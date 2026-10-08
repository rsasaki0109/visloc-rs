//! A bounded, blocking, drop-oldest queue for handing messages from DDS
//! receive threads to a processing thread.
//!
//! When the consumer falls behind, the oldest item is discarded so the
//! consumer always works on recent data and memory stays bounded.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

#[derive(Debug)]
pub struct DropOldestQueue<T> {
    state: Mutex<QueueState<T>>,
    ready: Condvar,
}

#[derive(Debug)]
struct QueueState<T> {
    items: VecDeque<T>,
    capacity: usize,
    dropped: u64,
    closed: bool,
}

impl<T> DropOldestQueue<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(QueueState {
                items: VecDeque::new(),
                capacity: capacity.max(1),
                dropped: 0,
                closed: false,
            }),
            ready: Condvar::new(),
        }
    }

    /// Enqueues `item`; returns `true` when an older item was dropped.
    pub fn push(&self, item: T) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.items.push_back(item);
        let mut dropped = false;
        while state.items.len() > state.capacity {
            state.items.pop_front();
            state.dropped += 1;
            dropped = true;
        }
        drop(state);
        self.ready.notify_one();
        dropped
    }

    /// Waits up to `timeout` for an item. `None` on timeout or when closed
    /// and drained.
    pub fn pop_timeout(&self, timeout: Duration) -> Option<T> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let (mut state, _) = self
            .ready
            .wait_timeout_while(state, timeout, |state| {
                state.items.is_empty() && !state.closed
            })
            .unwrap_or_else(|e| e.into_inner());
        state.items.pop_front()
    }

    /// Wakes all waiters; subsequent pops return remaining items then `None`.
    pub fn close(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).closed = true;
        self.ready.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).closed
    }

    pub fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .items
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dropped(&self) -> u64 {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn drops_oldest_when_full() {
        let queue = DropOldestQueue::new(2);
        assert!(!queue.push(1));
        assert!(!queue.push(2));
        assert!(queue.push(3));
        assert_eq!(queue.dropped(), 1);
        assert_eq!(queue.pop_timeout(Duration::ZERO), Some(2));
        assert_eq!(queue.pop_timeout(Duration::ZERO), Some(3));
        assert_eq!(queue.pop_timeout(Duration::from_millis(1)), None);
    }

    #[test]
    fn wakes_blocked_consumer() {
        let queue = Arc::new(DropOldestQueue::new(4));
        let producer = Arc::clone(&queue);
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            producer.push(7);
        });
        assert_eq!(queue.pop_timeout(Duration::from_secs(5)), Some(7));
        handle.join().unwrap();
        queue.close();
        assert!(queue.is_closed());
        assert_eq!(queue.pop_timeout(Duration::from_secs(5)), None);
    }
}
