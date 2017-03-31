package ohs.types.generic;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

//import fig.basic.Pair;

/**
 * Maintains counts of (key, value) pairs. The map is structured so that for every key, one can get a counter over values. Example usage:
 * keys might be words with values being POS tags, and the cnt being the number of occurences of that word/tag pair. The sub-counters
 * returned by getCounter(word) would be cnt distributions over tags for that word.
 * 
 * @author Dan Klein
 */
public class CounterMap<K, V> implements java.io.Serializable {
	private static final long serialVersionUID = 1L;

	public static void main(String[] args) {
		CounterMap<String, String> bigramCounterMap = new CounterMap<String, String>();
		bigramCounterMap.incrementCount("people", "run", 1);
		bigramCounterMap.incrementCount("cats", "growl", 2);
		bigramCounterMap.incrementCount("cats", "scamper", 3);
		System.out.println(bigramCounterMap);
		System.out.println("Entries for cats: " + bigramCounterMap.getCounter("cats"));
		System.out.println("Entries for dogs: " + bigramCounterMap.getCounter("dogs"));
		System.out.println("Count of cats scamper: " + bigramCounterMap.getCount("cats", "scamper"));
		System.out.println("Count of snakes slither: " + bigramCounterMap.getCount("snakes", "slither"));
		System.out.println("Total size: " + bigramCounterMap.totalSize());
		System.out.println("Total cnt: " + bigramCounterMap.totalCount());
		System.out.println(bigramCounterMap);
	}

	private Map<K, Counter<V>> cm;

	public CounterMap() {
		cm = new HashMap<K, Counter<V>>();
	}

	public CounterMap(CounterMap<K, V> cm) {
		this(cm.size());
		incrementAll(cm);
	}

	public CounterMap(int size) {
		cm = new HashMap<K, Counter<V>>(size);
	}

	/**
	 * Finds the key with maximum cnt. This is a linear operation, and ties are broken arbitrarily.
	 * 
	 * @return a key with minumum cnt
	 */
	public Pair<K, V> argMax() {
		double maxCount = Double.NEGATIVE_INFINITY;
		Pair<K, V> maxKey = null;
		for (Map.Entry<K, Counter<V>> entry : cm.entrySet()) {
			Counter<V> counter = entry.getValue();
			V localMax = counter.argMax();
			if (counter.getCount(localMax) > maxCount || maxKey == null) {
				maxKey = new Pair<K, V>(entry.getKey(), localMax);
				maxCount = counter.getCount(localMax);
			}
		}
		return maxKey;
	}

	public void clear() {
		clear(true);
	}

	public void clear(boolean deep_clear) {
		if (deep_clear) {
			for (Counter<V> c : cm.values()) {
				c.clear();
			}
		}
		cm.clear();
	}

	public boolean containKey(K key, V value) {
		Counter<V> c = cm.get(key);
		if (c == null) {
			return false;
		}
		return c.containsKey(value);
	}

	public boolean containsKey(K key) {
		return cm.containsKey(key);
	}

	protected Counter<V> ensureCounter(K key) {
		Counter<V> valueCounter = cm.get(key);
		if (valueCounter == null) {
			valueCounter = new Counter<V>();
			cm.put(key, valueCounter);
		}
		return valueCounter;
	}

	/**
	 * Gets the total cnt of the given key, or zero if that key is not present. Does not create any objs.
	 */
	public double getCount(K key) {
		Counter<V> valueCounter = cm.get(key);
		if (valueCounter == null)
			return 0.0;
		return valueCounter.totalCount();
	}

	/**
	 * Gets the cnt of the given (key, value) entry, or zero if that entry is not present. Does not create any objs.
	 */
	public double getCount(K key, V value) {
		Counter<V> valueCounter = cm.get(key);
		if (valueCounter == null)
			return 0.0;
		return valueCounter.getCount(value);
	}

	/**
	 * Gets the sub-counter for the given key. If there is none, a counter is created for that key, and installed in the CounterMap. You
	 * can, for example, add to the returned empty counter directly (though you shouldn't). This is so whether the key is present or not,
	 * modifying the returned counter has the same effect (but don't do it).
	 */
	public Counter<V> getCounter(K key) {
		return ensureCounter(key);
	}

	public Set<Map.Entry<K, Counter<V>>> getEntrySet() {
		return cm.entrySet();
	}

	public Counter<V> getInKeyCountSums() {
		Counter<V> ret = new Counter<V>();
		for (K key : cm.keySet()) {
			Counter<V> c = cm.get(key);
			for (Entry<V, Double> e : c.entrySet()) {
				ret.incrementCount(e.getKey(), e.getValue());
			}
		}
		return ret;
	}

	public Counter<K> getOutKeyCountSums() {
		Counter<K> ret = new Counter<K>();
		for (K key : cm.keySet()) {
			ret.setCount(key, cm.get(key).totalCount());
		}
		return ret;
	}

	public Iterator<Pair<K, V>> getPairIterator() {

		class PairIterator implements Iterator<Pair<K, V>> {

			Iterator<K> outerIt;
			Iterator<V> innerIt;
			K curKey;

			public PairIterator() {
				outerIt = keySet().iterator();
			}

			private boolean advance() {
				if (innerIt == null || !innerIt.hasNext()) {
					if (!outerIt.hasNext()) {
						return false;
					}
					curKey = outerIt.next();
					innerIt = getCounter(curKey).keySet().iterator();
				}
				return true;
			}

			@Override
			public boolean hasNext() {
				return advance();
			}

			@Override
			public Pair<K, V> next() {
				advance();
				assert curKey != null;
				return Pair.newPair(curKey, innerIt.next());
			}

			@Override
			public void remove() {
				// TODO Auto-generated method stub

			}

		}
		;
		return new PairIterator();
	}

	public void incrementAll(CounterMap<K, V> cMap) {
		for (Map.Entry<K, Counter<V>> entry : cMap.cm.entrySet()) {
			K key = entry.getKey();
			Counter<V> innerCounter = entry.getValue();
			for (Map.Entry<V, Double> innerEntry : innerCounter.entrySet()) {
				V value = innerEntry.getKey();
				incrementCount(key, value, innerEntry.getValue());
			}
		}
	}

	public void incrementAll(CounterMap<K, V> cMap, double scaleFactor) {
		for (Map.Entry<K, Counter<V>> entry : cMap.cm.entrySet()) {
			K key = entry.getKey();
			Counter<V> innerCounter = entry.getValue();
			for (Map.Entry<V, Double> innerEntry : innerCounter.entrySet()) {
				V value = innerEntry.getKey();
				incrementCount(key, value, scaleFactor * innerEntry.getValue());
			}
		}
	}

	public void incrementAll(Map<K, V> map, double count) {
		for (Map.Entry<K, V> entry : map.entrySet()) {
			incrementCount(entry.getKey(), entry.getValue(), count);
		}
	}

	/**
	 * Increments the cnt for a particular (key, value) pair.
	 */
	public void incrementCount(K key, V value, double count) {
		Counter<V> valueCounter = ensureCounter(key);
		valueCounter.incrementCount(value, count);
	}

	public String info() {
		StringBuffer sb = new StringBuffer();

		int min_keys = Integer.MAX_VALUE;
		int max_keys = -Integer.MIN_VALUE;
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		double sum = 0;
		int cnt = 0;
		for (K key1 : keySet()) {
			Counter<V> c = getCounter(key1);
			max = Math.max(max, c.max());
			min = Math.min(min, c.min());
			max_keys = Math.max(max_keys, c.size());
			min_keys = Math.min(min_keys, c.size());
			sum += c.totalCount();
			cnt += c.size();
		}

		sb.append(String.format("outer keys:\t%d\n", size()));
		sb.append(String.format("max inner keys:\t%d\n", max_keys));
		sb.append(String.format("min inner keys:\t%d\n", min_keys));
		sb.append(String.format("max:\t%f\n", max));
		sb.append(String.format("min:\t%f\n", min));
		sb.append(String.format("avg:\t%f\n", sum / cnt));
		return sb.toString();
	}

	public Set<V> innerKeySet() {
		Set<V> ret = new HashSet<V>();
		for (Counter<V> c : cm.values()) {
			ret.addAll(c.keySet());
		}
		return ret;
	}

	/**
	 * Constructs reverse CounterMap where the cnt of a pair (k,v) is the cnt of (v,k) in the current CounterMap
	 * 
	 * @return
	 */
	public CounterMap<V, K> invert() {
		CounterMap<V, K> invertCounterMap = new CounterMap<V, K>();
		for (K key : this.keySet()) {
			Counter<V> keyCounts = this.getCounter(key);
			for (V val : keyCounts.keySet()) {
				double count = keyCounts.getCount(val);
				invertCounterMap.setCount(val, key, count);
			}
		}
		invertCounterMap.trimToSize();
		return invertCounterMap;
	}

	/**
	 * True if there are no ents in the CounterMap (false does not mean totalCount > 0)
	 */
	public boolean isEmpty() {
		return size() == 0;
	}

	public boolean isEqualTo(CounterMap<K, V> map) {
		boolean tmp = true;
		CounterMap<K, V> bigger = map.size() > size() ? map : this;
		for (K k : bigger.keySet()) {
			tmp &= map.getCounter(k).isEqualTo(getCounter(k));
		}
		return tmp;
	}

	/**
	 * Returns the keys that have been inserted into this CounterMap.
	 */
	public Set<K> keySet() {
		return cm.keySet();
	}

	public Set<V> keySetOfCounter(K key) {
		return cm.get(key).keySet();
	}

	public void normalize() {
		for (K key : keySet()) {
			getCounter(key).normalize();
		}
	}

	public void normalizeWithDiscount(double discount) {
		for (K key : keySet()) {
			Counter<V> ctr = getCounter(key);
			double totalCount = ctr.totalCount();
			for (V value : ctr.keySet()) {
				ctr.setCount(value, (ctr.getCount(value) - discount) / totalCount);
			}
		}
	}

	public void prune(final Set<K> toRemove) {
		for (final K e : toRemove) {
			removeKey(e);
		}
	}

	public void pruneExcept(final Set<K> toKeep) {
		final List<K> toRemove = new ArrayList<K>();
		for (final K key : cm.keySet()) {
			if (!toKeep.contains(key))
				toRemove.add(key);
		}
		for (final K e : toRemove) {
			removeKey(e);
		}
	}

	public Counter<V> removeKey(K oldIndex) {
		return cm.remove(oldIndex);

	}

	/**
	 * Scale all ents in <code>CounterMap</code> by <code>scaleFactor</code>
	 * 
	 * @param scaleFactor
	 */
	public void scale(double scaleFactor) {
		for (K key : keySet()) {
			Counter<V> counts = getCounter(key);
			counts.scale(scaleFactor);
		}
	}

	/**
	 * Sets the cnt for a particular (key, value) pair.
	 */
	public void setCount(K key, V value, double count) {
		Counter<V> valueCounter = ensureCounter(key);
		valueCounter.setCount(value, count);
	}

	public void setCounter(K newIndex, Counter<V> counter) {
		cm.put(newIndex, counter);

	}

	/**
	 * The number of keys in this CounterMap (not the number of key-value ents -- use totalSize() for that)
	 */
	public int size() {
		return cm.size();
	}

	public int sizeOfCounter(K key) {
		return cm.get(key).size();
	}

	public int sizeOfMaxInnerCounter() {
		int ret = 0;
		for (Counter<V> c : cm.values()) {
			if (ret < c.size()) {
				ret = c.size();
			}
		}
		return ret;
	}

	@Override
	public String toString() {
		return toString(50, 50);
	}

	public String toString(int row_size, int col_size) {
		StringBuilder sb = new StringBuilder("[\n");
		int numKeys = 0;

		for (K key : getOutKeyCountSums().getSortedKeys()) {
			Counter<V> inner = cm.get(key);
			if (++numKeys > row_size) {
				sb.append("...\n");
				break;
			}

			double sum = inner.totalCount();
			int tmp = (int) sum;

			if (sum - tmp == 0) {
				sb.append(String.format("   %s:%d -> ", key, tmp));
			} else {
				sb.append(String.format("   %s:%f -> ", key, sum));
			}
			sb.append(inner.toStringSortedByValues(true, false, col_size, " "));
			sb.append("\n");
		}
		sb.append("]");
		return sb.toString();
	}

	/**
	 * Returns the total of all counts in sub-counters. This implementation is linear; it recalculates the total each time.
	 */
	public double totalCount() {
		double total = 0.0;
		for (Map.Entry<K, Counter<V>> entry : cm.entrySet()) {
			Counter<V> counter = entry.getValue();
			total += counter.totalCount();
		}
		return total;
	}

	/**
	 * Returns the total number of (key, value) ents in the CounterMap (not their total counts).
	 */
	public int totalSize() {
		int total = 0;
		for (Map.Entry<K, Counter<V>> entry : cm.entrySet()) {
			Counter<V> counter = entry.getValue();
			total += counter.size();
		}
		return total;
	}

	public void trimToSize() {
		Map<K, Counter<V>> cm = new HashMap<K, Counter<V>>();
		for (Entry<K, Counter<V>> e : this.cm.entrySet()) {
			K k = e.getKey();
			Counter<V> v = e.getValue();
			v.trimToSize();
			if (v.size() > 0) {
				cm.put(k, v);
			}
		}
		this.cm = cm;

	}

}
