/*
    SMetric: session-centric, SLO-aware routing for agentic workloads

    Agent sessions resend their whole history on every turn, so each follow-up
    request can reuse the KV cache its previous turn left on one worker. SMetric
    keeps a session on that worker while it can still meet the request's TTFT
    SLO, and otherwise migrates the session to the least-loaded worker. It keeps
    no per-session state: the turn comes from the request's messages and the
    session's worker is the one with the longest cached prefix.

    Per request, with c[i] the cached prefix on worker i and q[i] the prefill
    cost queued there:

        prev = argmax(c)
        if turn > 0 and c[prev] > HIT_RATIO * history
                and (meets_ttft(prev) or no worker meets_ttft):
            route to prev                                       # stick
        else:
            route to rr_argmin(load(i) * in_flight_requests(i))  # balance

        load(i)       = q[i] + prefill_cost(req, c[i])
        meets_ttft(i) = load(i) / prefill_rate(i) <= SLACK * ttft_slo(req)
        prefill_cost  = n + n * (L - n / 2) / L_eq     # n new of L prompt tokens

    `history` is what the previous turn sent: the prompt up to its latest
    assistant message. A cached prefix much shorter than that means the worker
    has evicted the session, which then routes like a new one.

    The router only sees text, so prompt, hit and history lengths are the
    approximate prefix tree's character counts divided by `CHARS_PER_TOKEN`.

    `prefill_rate(i)` is measured, not configured. Each request contributes its
    prefill cost over its time to first token; TTFT includes queueing, so the
    fastest of these observations approach the worker's real prefill speed.
    Until a worker has enough streamed samples, its TTFT check passes and the
    policy routes as plain session-centric scheduling.
*/

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders, RequestTracker};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::tree::Tree;
use dashmap::DashMap;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tracing::debug;

/// Characters per token, the estimate the router uses wherever it lacks a tokenizer.
const CHARS_PER_TOKEN: f64 = 4.0;
/// Number of recent requests per worker the prefill rate is measured over.
const PREFILL_RATE_WINDOW: usize = 64;
/// Samples a worker needs before its measured prefill rate is used.
const PREFILL_RATE_MIN_SAMPLES: usize = 8;
/// Quantile of the per-request rates taken as the worker's prefill rate. The
/// least-queued requests are the fastest; the maximum would be a single outlier.
const PREFILL_RATE_QUANTILE: f64 = 0.9;

/// Configuration for the SMetric policy
#[derive(Debug, Clone)]
pub struct SMetricConfig {
    /// A worker still holds the session if its cached prefix exceeds this
    /// fraction of the history the previous turn sent.
    pub hit_ratio: f64,
    /// Scales the TTFT SLO a session's worker must meet to keep the session.
    pub slack: f64,
    /// Fixed part of the TTFT SLO, in seconds.
    pub ttft_slo_secs: f64,
    /// Part of the TTFT SLO that grows with prompt length, in seconds per 1K tokens.
    pub ttft_slo_secs_per_1k_tokens: f64,
    /// Context length at which a new token's attention costs as much as its
    /// linear layers, about active_params / (2 * layers * query_heads * head_dim).
    /// `None` prices prefill by new tokens alone.
    pub attention_crossover_tokens: Option<f64>,
    /// Interval between LRU eviction cycles of the prefix tree.
    pub eviction_interval_secs: u64,
    /// Maximum characters the prefix tree keeps per worker.
    pub max_tree_size: usize,
}

impl Default for SMetricConfig {
    fn default() -> Self {
        Self {
            hit_ratio: 0.5,
            slack: 1.0,
            ttft_slo_secs: 1.0,
            ttft_slo_secs_per_1k_tokens: 0.0625,
            attention_crossover_tokens: None,
            eviction_interval_secs: 120,
            max_tree_size: 1 << 26,
        }
    }
}

/// Session-centric, SLO-aware routing policy
#[derive(Debug)]
pub struct SMetricPolicy {
    config: SMetricConfig,
    tree: Arc<Tree>,
    workers: DashMap<String, Arc<Mutex<WorkerPrefill>>>,
    tie_breaker: AtomicUsize,
}

impl SMetricPolicy {
    pub fn new() -> Self {
        Self::with_config(SMetricConfig::default())
    }

    pub fn with_config(config: SMetricConfig) -> Self {
        let tree = Arc::new(Tree::new());
        if config.eviction_interval_secs > 0 {
            let tree = Arc::downgrade(&tree);
            let interval = Duration::from_secs(config.eviction_interval_secs);
            let max_tree_size = config.max_tree_size;
            thread::spawn(move || loop {
                thread::sleep(interval);
                let Some(tree) = tree.upgrade() else { break };
                tree.evict_tenant_by_size(max_tree_size);
            });
        }
        Self {
            config,
            tree,
            workers: DashMap::new(),
            tie_breaker: AtomicUsize::new(0),
        }
    }

    fn worker_prefill(&self, url: &str) -> Arc<Mutex<WorkerPrefill>> {
        if let Some(prefill) = self.workers.get(url) {
            return Arc::clone(prefill.value());
        }
        Arc::clone(self.workers.entry(url.to_string()).or_default().value())
    }

    /// Prefill cost of a `len`-token prompt with `hit` tokens cached, in
    /// linear-token equivalents: each new token attends to ~`len - new/2` others.
    fn prefill_cost(&self, len: f64, hit: f64) -> f64 {
        let new = (len - hit).max(0.0);
        match self.config.attention_crossover_tokens {
            Some(crossover) => new + new * (len - new / 2.0) / crossover,
            None => new,
        }
    }

    fn ttft_budget_secs(&self, len: f64) -> f64 {
        self.config.slack
            * (self.config.ttft_slo_secs + self.config.ttft_slo_secs_per_1k_tokens * len / 1000.0)
    }
}

impl Default for SMetricPolicy {
    fn default() -> Self {
        Self::new()
    }
}

struct Candidate {
    idx: usize,
    hit: f64,
    cost: f64,
    /// Prefill cost queued on the worker plus this request's own.
    load: f64,
    prefill_rate: Option<f64>,
}

impl LoadBalancingPolicy for SMetricPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.select_worker_tracked(workers, request_text, headers)
            .map(|(idx, _)| idx)
    }

    fn select_worker_tracked(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<(usize, Option<Box<dyn RequestTracker>>)> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        let prompt = SessionPrompt::parse(request_text.unwrap_or(""));
        let len = prompt.chars as f64 / CHARS_PER_TOKEN;
        let history = prompt.history_chars as f64 / CHARS_PER_TOKEN;
        let candidates: Vec<Candidate> = healthy_indices
            .iter()
            .map(|&idx| {
                let url = workers[idx].url();
                let hit = self.tree.prefix_match_tenant_char_count(&prompt.text, url) as f64
                    / CHARS_PER_TOKEN;
                let cost = self.prefill_cost(len, hit);
                let prefill = self.worker_prefill(url);
                let prefill = prefill.lock();
                Candidate {
                    idx,
                    hit,
                    cost,
                    load: prefill.queued_cost + cost,
                    prefill_rate: prefill.prefill_rate,
                }
            })
            .collect();

        let budget = self.ttft_budget_secs(len);
        let meets_ttft = |candidate: &Candidate| {
            candidate
                .prefill_rate
                .is_none_or(|rate| candidate.load / rate <= budget)
        };
        let prev = candidates
            .iter()
            .max_by(|a, b| a.hit.total_cmp(&b.hit).then(b.load.total_cmp(&a.load)))?;
        let stick = prompt.turn > 0
            && prev.hit > self.config.hit_ratio * history
            && (meets_ttft(prev) || !candidates.iter().any(meets_ttft));

        let selected = if stick {
            prev
        } else {
            let score =
                |candidate: &Candidate| candidate.load * workers[candidate.idx].load() as f64;
            let best = candidates.iter().map(score).fold(f64::INFINITY, f64::min);
            let tied: Vec<&Candidate> = candidates.iter().filter(|c| score(c) == best).collect();
            tied[self.tie_breaker.fetch_add(1, Ordering::Relaxed) % tied.len()]
        };

        let url = workers[selected.idx].url();
        debug!(
            worker = url,
            turn = prompt.turn,
            hit_tokens = selected.hit,
            history_tokens = history,
            stick,
            "smetric decision"
        );
        if !prompt.text.is_empty() {
            self.tree.insert(&prompt.text, url);
        }
        RouterMetrics::record_processed_request(url);
        RouterMetrics::record_policy_decision(self.name(), url);

        let tracker = PrefillTracker::dispatch(self.worker_prefill(url), selected.cost);
        Some((selected.idx, Some(Box::new(tracker))))
    }

    fn name(&self) -> &'static str {
        "smetric"
    }

    fn needs_request_body(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The part of a request SMetric routes on, read from its JSON body.
#[derive(Debug, Default, PartialEq)]
struct SessionPrompt {
    /// Tool schemas followed by every message, in order. Agents only append to
    /// this, so each turn extends the previous turn's text.
    text: String,
    chars: usize,
    /// Characters of `text` before the latest assistant message: what the
    /// previous turn sent.
    history_chars: usize,
    /// Assistant messages in the conversation; 0 is a session's first request.
    turn: usize,
}

impl SessionPrompt {
    fn parse(body: &str) -> Self {
        let value = serde_json::from_str::<Value>(body).unwrap_or(Value::Null);
        if let Some(messages) = value.get("messages").and_then(Value::as_array) {
            let mut prompt = Self::default();
            if let Some(tools) = value.get("tools") {
                prompt.push(&tools.to_string());
            }
            for message in messages {
                if message.get("role").and_then(Value::as_str) == Some("assistant") {
                    prompt.history_chars = prompt.chars;
                    prompt.turn += 1;
                }
                prompt.push(&message.to_string());
            }
            return prompt;
        }
        // A completion prompt carries no turns; route it as a session's first request.
        let mut prompt = Self::default();
        match value.get("prompt") {
            Some(Value::String(text)) => prompt.push(text),
            Some(_) => {}
            None if value.is_null() => prompt.push(body),
            None => {}
        }
        prompt
    }

    fn push(&mut self, text: &str) {
        self.text.push_str(text);
        self.chars += text.chars().count();
    }
}

/// Prefill feedback of one worker.
#[derive(Debug, Default)]
struct WorkerPrefill {
    /// Prefill cost of the requests dispatched here that have no first token yet.
    queued_cost: f64,
    queued_requests: usize,
    /// Recent per-request prefill rates (cost per second to first token).
    samples: VecDeque<f64>,
    prefill_rate: Option<f64>,
}

impl WorkerPrefill {
    fn dequeue(&mut self, cost: f64) {
        self.queued_requests -= 1;
        self.queued_cost = if self.queued_requests == 0 {
            0.0
        } else {
            (self.queued_cost - cost).max(0.0)
        };
    }

    fn record_rate(&mut self, rate: f64) {
        if self.samples.len() == PREFILL_RATE_WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(rate);
        if self.samples.len() >= PREFILL_RATE_MIN_SAMPLES {
            let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
            sorted.sort_by(f64::total_cmp);
            let rank = PREFILL_RATE_QUANTILE * (sorted.len() - 1) as f64;
            self.prefill_rate = Some(sorted[rank.round() as usize]);
        }
    }
}

/// One request's prefill on a worker: queued until its first token arrives or it
/// leaves the worker. Only a streamed first token yields a rate sample.
struct PrefillTracker {
    worker: Arc<Mutex<WorkerPrefill>>,
    cost: f64,
    dispatched_at: Instant,
    queued: bool,
}

impl PrefillTracker {
    fn dispatch(worker: Arc<Mutex<WorkerPrefill>>, cost: f64) -> Self {
        {
            let mut prefill = worker.lock();
            prefill.queued_cost += cost;
            prefill.queued_requests += 1;
        }
        Self {
            worker,
            cost,
            dispatched_at: Instant::now(),
            queued: true,
        }
    }
}

impl RequestTracker for PrefillTracker {
    fn on_first_token(&mut self) {
        if !std::mem::replace(&mut self.queued, false) {
            return;
        }
        let ttft = self.dispatched_at.elapsed().as_secs_f64();
        let mut prefill = self.worker.lock();
        prefill.dequeue(self.cost);
        if self.cost > 0.0 && ttft > 0.0 {
            prefill.record_rate(self.cost / ttft);
        }
    }
}

impl Drop for PrefillTracker {
    fn drop(&mut self) {
        if self.queued {
            self.worker.lock().dequeue(self.cost);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use serde_json::json;

    fn policy() -> SMetricPolicy {
        SMetricPolicy::with_config(SMetricConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        })
    }

    fn workers(n: usize) -> Vec<Arc<dyn Worker>> {
        (0..n)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://w{i}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect()
    }

    fn chat(turns: usize) -> String {
        let mut messages = vec![json!({"role": "system", "content": "You are a coding agent."})];
        for turn in 0..=turns {
            if turn > 0 {
                messages.push(json!({"role": "assistant", "content": format!("reply {turn}")}));
            }
            messages
                .push(json!({"role": "user", "content": format!("{turn}: {}", "x".repeat(400))}));
        }
        json!({"model": "m", "messages": messages}).to_string()
    }

    /// Route `body` and keep its prefill queued, as a dispatched request does.
    fn dispatch(
        policy: &SMetricPolicy,
        workers: &[Arc<dyn Worker>],
        body: &str,
    ) -> (usize, Box<dyn RequestTracker>) {
        let (idx, tracker) = policy
            .select_worker_tracked(workers, Some(body), None)
            .unwrap();
        (idx, tracker.unwrap())
    }

    fn set_prefill_rate(policy: &SMetricPolicy, url: &str, rate: f64) {
        let prefill = policy.worker_prefill(url);
        let mut prefill = prefill.lock();
        for _ in 0..PREFILL_RATE_MIN_SAMPLES {
            prefill.record_rate(rate);
        }
    }

    #[test]
    fn session_prompt_reads_turn_and_history() {
        let first = SessionPrompt::parse(&chat(0));
        assert_eq!(first.turn, 0);

        let third = SessionPrompt::parse(&chat(2));
        assert_eq!(third.turn, 2);
        let previous = SessionPrompt::parse(&chat(1));
        assert!(third.text.starts_with(&previous.text));
        assert_eq!(third.history_chars, previous.chars);

        let completion = SessionPrompt::parse(r#"{"prompt": "hello"}"#);
        assert_eq!((completion.text.as_str(), completion.turn), ("hello", 0));
    }

    #[test]
    fn first_turns_balance_and_follow_ups_stick() {
        let policy = policy();
        let workers = workers(2);
        let (home, _session) = dispatch(&policy, &workers, &chat(0));
        workers[home].increment_load();

        // Another session's first turn avoids the busy worker.
        let other_session = chat(0).replace("0: ", "other: ");
        assert_ne!(dispatch(&policy, &workers, &other_session).0, home);

        // The first session's follow-up returns to its cache despite the load.
        assert_eq!(dispatch(&policy, &workers, &chat(1)).0, home);
    }

    #[test]
    fn evicted_session_routes_as_new() {
        let policy = policy();
        let workers = workers(2);
        let (home, _session) = dispatch(&policy, &workers, &chat(0));
        workers[home].increment_load();
        policy.tree.remove_tenant(workers[home].url());
        assert_ne!(dispatch(&policy, &workers, &chat(1)).0, home);
    }

    #[test]
    fn session_migrates_only_when_another_worker_meets_ttft() {
        let policy = policy();
        let workers = workers(2);
        let (home, _first_turn) = dispatch(&policy, &workers, &chat(0));
        workers[home].increment_load();
        let other = 1 - home;

        // No worker prefills fast enough to meet the SLO: keep the cache.
        set_prefill_rate(&policy, workers[home].url(), 1.0);
        set_prefill_rate(&policy, workers[other].url(), 1.0);
        assert_eq!(dispatch(&policy, &workers, &chat(1)).0, home);

        set_prefill_rate(&policy, workers[other].url(), 1e9);
        assert_eq!(dispatch(&policy, &workers, &chat(1)).0, other);
    }

    #[test]
    fn tracker_dequeues_and_measures_prefill_rate() {
        let policy = policy();
        let workers = workers(1);
        let prefill = policy.worker_prefill(workers[0].url());

        let mut trackers: Vec<_> = (0..PREFILL_RATE_MIN_SAMPLES)
            .map(|i| dispatch(&policy, &workers, &chat(i)).1)
            .collect();
        assert!(prefill.lock().queued_cost > 0.0);

        thread::sleep(Duration::from_millis(1));
        for tracker in &mut trackers {
            tracker.on_first_token();
        }
        assert_eq!(prefill.lock().queued_cost, 0.0);
        assert!(prefill.lock().prefill_rate.is_some());

        let unfinished = dispatch(&policy, &workers, &chat(0)).1;
        drop(unfinished);
        assert_eq!(prefill.lock().queued_requests, 0);
    }
}
