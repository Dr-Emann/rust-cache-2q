# 0.10.0

## Added
* `VacantEntry::is_remembered()` to test if a vacant entry is remembered, and will become a 
frequent entry if inserted

## Changed
* `Cache::get()` removed
* `Cache::get_mut()` renamed to `Cache::get()`
* `Cache::new(size)` frequent size and recent size both are `size`, and `size * 4` ghosts are kept

# 0.9.0

## Changed
* Use Linked Hashmaps to prevent linear lookups. This adds a `Hash` generic bound in a lot of places

# 0.8.4

## Changed
* The minimum size of a Cache is now 1, instead of 2

# 0.8.3

## Added
* peek() function to Cache, to allow getting an item without requiring mutable access

# 0.8.2

## Added
* Changelog added
