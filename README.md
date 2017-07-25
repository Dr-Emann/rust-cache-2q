# Two Queue (2Q) Cache

A 2Q Cache which maps keys to values

2Q is an enhancement over an LRU cache by tracking both recent and frequently accessed entries
separately. This avoids the cache being trashed by a scan of many new items: Only the recent
list will be trashed.

The cache is split into 3 sections, recent entries, frequent entries, and ghost entries
recent contains the most recently added entries.
frequent is an LRU cache which contains entries which are frequently accessed
ghost contains the keys which have been recently evicted from the recent cache.

New entries in the cache are initially placed in recent.
After recent fills up, the oldest entry from recent will be removed, and its key is placed in
ghost. When an entry is requested and not found, but its key is found in the ghost list,
an entry is pushed to the front of frequent.

# Examples

```rust
use cache_2q::Cache;

// type inference lets us omit an explicit type signature (which
// would be `Cache<&str, &str>` in this example).
let mut book_reviews = Cache::new(1024);

// review some books.
book_reviews.insert("Adventures of Huckleberry Finn",    "My favorite book.");
book_reviews.insert("Grimms' Fairy Tales",               "Masterpiece.");
book_reviews.insert("Pride and Prejudice",               "Very enjoyable.");
book_reviews.insert("The Adventures of Sherlock Holmes", "Eye lyked it alot.");

// check for a specific one.
if !book_reviews.contains_key("Les Misérables") {
    println!("We've got {} reviews, but Les Misérables ain't one.",
             book_reviews.len());
}

// oops, this review has a lot of spelling mistakes, let's delete it.
book_reviews.remove("The Adventures of Sherlock Holmes");

// look up the values associated with some keys.
let to_find = ["Pride and Prejudice", "Alice's Adventure in Wonderland"];
for book in &to_find {
    match book_reviews.get(book) {
        Some(review) => println!("{}: {}", book, review),
        None => println!("{} is unreviewed.", book)
    }
}

// iterate over everything.
for (book, review) in &book_reviews {
    println!("{}: \"{}\"", book, review);
}
```

Cache also implements an Entry API, which allows for more complex methods of getting,
setting, updating and removing keys and their values:

```rust
use cache_2q::Cache;

// type inference lets us omit an explicit type signature (which
// would be `Cache<&str, u8>` in this example).
let mut player_stats = Cache::new(32);

fn random_stat_buff() -> u8 {
    // could actually return some random value here - let's just return
    // some fixed value for now
    42
}

// insert a key only if it doesn't already exist
player_stats.entry("health").or_insert(100);

// insert a key using a function that provides a new value only if it
// doesn't already exist
player_stats.entry("defence").or_insert_with(random_stat_buff);

// update a key, guarding against the key possibly not being set
let stat = player_stats.entry("attack").or_insert(100);
*stat += random_stat_buff();
```