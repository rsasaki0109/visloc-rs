//! [`IdMap`]: the reconstruction's id-keyed tables (images, frames,
//! 3D points) as a vector indexed by id.
//!
//! COLMAP's ids are small dense integers (database row ids, and 3D point ids
//! handed out by a counter), so indexing a vector replaces a `BTreeMap`
//! search — the mapper looks ids up millions of times per image. Iteration
//! still runs in ascending id order, exactly like the `BTreeMap` it
//! replaces, so every loop over a table sees the same sequence.

use std::fmt;

/// Ids at or past this many slots are rejected (a vector this long would be
/// a corrupt id, not a reconstruction).
const MAX_SLOTS: u64 = 1 << 32;

/// A map from `u64` ids to `V`, stored densely by id; iteration is in
/// ascending id order.
#[derive(Clone)]
pub struct IdMap<V> {
    slots: Vec<Option<(u64, V)>>,
    len: usize,
}

impl<V> Default for IdMap<V> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            len: 0,
        }
    }
}

impl<V> IdMap<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn get(&self, id: &u64) -> Option<&V> {
        match self.slots.get(usize::try_from(*id).ok()?) {
            Some(Some((_, v))) => Some(v),
            _ => None,
        }
    }

    #[inline]
    pub fn get_mut(&mut self, id: &u64) -> Option<&mut V> {
        match self.slots.get_mut(usize::try_from(*id).ok()?) {
            Some(Some((_, v))) => Some(v),
            _ => None,
        }
    }

    #[inline]
    pub fn contains_key(&self, id: &u64) -> bool {
        self.get(id).is_some()
    }

    /// Inserts `value` at `id`, returning the value it replaced.
    pub fn insert(&mut self, id: u64, value: V) -> Option<V> {
        assert!(id < MAX_SLOTS, "id {id} is too large for an IdMap");
        let index = id as usize;
        if index >= self.slots.len() {
            self.slots.resize_with(index + 1, || None);
        }
        let previous = self.slots[index].replace((id, value)).map(|(_, v)| v);
        if previous.is_none() {
            self.len += 1;
        }
        previous
    }

    pub fn remove(&mut self, id: &u64) -> Option<V> {
        let slot = self.slots.get_mut(usize::try_from(*id).ok()?)?;
        let removed = slot.take().map(|(_, v)| v);
        if removed.is_some() {
            self.len -= 1;
        }
        removed
    }

    /// `(id, value)` pairs in ascending id order.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&u64, &V)> + Clone {
        self.slots
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(id, v)| (id, v)))
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (&u64, &mut V)> {
        self.slots
            .iter_mut()
            .filter_map(|slot| slot.as_mut().map(|(id, v)| (&*id, v)))
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &u64> + Clone {
        self.iter().map(|(id, _)| id)
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> + Clone {
        self.iter().map(|(_, v)| v)
    }

    pub fn values_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut V> {
        self.iter_mut().map(|(_, v)| v)
    }

    /// The entry with the smallest id.
    pub fn first_key_value(&self) -> Option<(&u64, &V)> {
        self.iter().next()
    }

    /// The entry with the largest id.
    pub fn last_key_value(&self) -> Option<(&u64, &V)> {
        self.iter().next_back()
    }
}

impl<V: PartialEq> PartialEq for IdMap<V> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<V: fmt::Debug> fmt::Debug for IdMap<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<V> std::ops::Index<&u64> for IdMap<V> {
    type Output = V;

    fn index(&self, id: &u64) -> &V {
        self.get(id)
            .unwrap_or_else(|| panic!("id {id} is not in the IdMap"))
    }
}

impl<'a, V> IntoIterator for &'a IdMap<V> {
    type Item = (&'a u64, &'a V);
    type IntoIter = Box<dyn DoubleEndedIterator<Item = (&'a u64, &'a V)> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}

impl<V> FromIterator<(u64, V)> for IdMap<V> {
    fn from_iter<I: IntoIterator<Item = (u64, V)>>(iter: I) -> Self {
        let mut map = Self::new();
        for (id, v) in iter {
            map.insert(id, v);
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn behaves_like_a_btreemap() {
        let mut ours = IdMap::new();
        let mut reference = BTreeMap::new();
        let mut state = 12345u64;
        for step in 0..2000u64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let id = (state >> 33) % 300;
            if state & 3 == 0 {
                assert_eq!(ours.remove(&id), reference.remove(&id));
            } else {
                assert_eq!(ours.insert(id, step), reference.insert(id, step));
            }
            assert_eq!(ours.len(), reference.len());
            assert_eq!(ours.get(&id), reference.get(&id));
        }
        assert!(ours.iter().eq(reference.iter()));
        assert!(ours.iter().rev().eq(reference.iter().rev()));
        assert_eq!(ours.first_key_value(), reference.first_key_value());
        assert_eq!(ours.last_key_value(), reference.last_key_value());
        assert!(!ours.contains_key(&u64::MAX));
    }
}
