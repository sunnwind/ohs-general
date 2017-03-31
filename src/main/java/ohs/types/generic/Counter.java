package ohs.types.generic;

import java.io.Serializable;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

/**
 * A map from objs to doubles. Includes convenience methods for getting, setting, and incrementing element counts. Objects not in the
 * counter will return a cnt of zero. The counter is backed by a HashMap .(unless specified otherwise with the MapFactory constructor).
 * 
 * @author lots of people
 */
public class Counter<E> implements Serializable {
	// Compare by value.
	public class EntryValueComparator implements Comparator<Entry<E, Double>> {

		private final boolean descending;

		/**
		 * @param descending
		 */
		public EntryValueComparator(final boolean descending) {
			super();
			this.descending = descending;
		}

		@Override
		public int compare(final Entry<E, Double> e1, final Entry<E, Double> e2) {
			return descending ? Double.compare(e2.getValue(), e1.getValue()) : Double.compare(e1.getValue(), e2.getValue());
		}
	}

	private static final long serialVersionUID = 1L;

	public static <L> Counter<L> absCounts(final Counter<L> counts) {
		final Counter<L> res = new Counter<L>();
		for (final Map.Entry<L, Double> entry : counts.entrySet()) {
			res.incrementCount(entry.getKey(), Math.abs(entry.getValue()));
		}
		return res;
	}

	public static void main(final String[] args) throws Exception {
		final Counter<String> counter = new Counter<String>();
		System.out.println(counter);
		counter.incrementCount("planets", 7);
		System.out.println(counter);
		counter.incrementCount("planets", 1);
		System.out.println(counter);
		counter.setCount("suns", 1);
		System.out.println(counter);
		counter.setCount("aliens", 0);
		System.out.println(counter);
		System.out.println(counter.toString());
		System.out.println("Total: " + counter.totalCount());

	}

	private Map<E, Double> entries;

	private boolean dirty = true;

	private double cacheTotal = 0.0;

	private double defaultCount = 0.0;

	public Counter() {
		entries = new HashMap<E, Double>();
	}

	public Counter(final Collection<? extends E> collection) {
		this(collection.size());
		incrementAll(collection, 1.0);
	}

	public Counter(final Counter<? extends E> counter) {
		this(counter.size());
		incrementAll(counter);
	}

	public Counter(final E[] items) {
		this(items.length);
		incrementAll(items);
	}

	public Counter(int size) {
		entries = new HashMap<E, Double>(size);
	}

	public boolean approxEquals(final Counter<E> other, final double tol) {
		for (final E key : keySet()) {
			if (Math.abs(getCount(key) - other.getCount(key)) > tol) {//
				return false;
			}
		}
		for (final E key : other.keySet()) {
			if (Math.abs(getCount(key) - other.getCount(key)) > tol) {
				//
				return false;
			}
		}
		return true;
	}

	/**
	 * Finds the key with maximum cnt. This is a linear operation, and ties are broken arbitrarily.
	 * 
	 * @return a key with minumum cnt
	 */
	public E argMax() {
		double maxCount = Double.NEGATIVE_INFINITY;
		E maxKey = null;
		for (final Map.Entry<E, Double> entry : entries.entrySet()) {
			if (entry.getValue() > maxCount || maxKey == null) {
				maxKey = entry.getKey();
				maxCount = entry.getValue();
			}
		}
		return maxKey;
	}

	public E argMin() {
		double minCount = Double.POSITIVE_INFINITY;
		E minKey = null;
		for (final Map.Entry<E, Double> entry : entries.entrySet()) {
			if (entry.getValue() < minCount || minKey == null) {
				minKey = entry.getKey();
				minCount = entry.getValue();
			}
		}
		return minKey;
	}

	public double average() {
		double ret = 0;
		ret = totalCount() / size();
		return ret;
	}

	public void clear() {
		entries.clear();
		dirty = true;
	}

	/**
	 * Returns whether the counter contains the given key. Note that this is the way to distinguish keys which are in the counter with cnt
	 * zero, and those which are not in the counter (and will therefore return cnt zero from getCount().
	 * 
	 * @param key
	 * @return whether the counter contains the key
	 */
	public boolean containsKey(final E key) {
		return entries.containsKey(key);
	}

	public Counter<E> difference(final Counter<E> counter) {
		final Counter<E> clone = new Counter<E>(this);
		for (final E key : counter.keySet()) {
			final double count = counter.getCount(key);
			clone.incrementCount(key, -1 * count);
		}
		return clone;
	}

	public double dotProduct(final Counter<E> other) {
		double sum = 0.0;

		Counter<E> small = this;
		Counter<E> large = other;

		if (small.size() > large.size()) {
			small = other;
			large = this;
		}

		for (final Map.Entry<E, Double> entry : small.getEntrySet()) {
			final double otherCount = large.getCount(entry.getKey());
			if (otherCount == 0.0)
				continue;
			final double value = entry.getValue();
			if (value == 0.0)
				continue;
			sum += value * otherCount;

		}
		return sum;
	}

	public Set<Entry<E, Double>> entrySet() {
		return entries.entrySet();
	}

	/**
	 * Get the cnt of the element, or zero if the element is not in the counter.
	 * 
	 * @param key
	 * @return
	 */
	public double getCount(final E key) {
		final Double value = entries.get(key);
		if (value == null)
			return defaultCount;
		return value;
	}

	public double getDefaultCount() {
		return defaultCount;
	}

	public Collection<Entry<E, Double>> getEntriesSortedByDecreasingCount() {
		final List<Entry<E, Double>> sorted = new ArrayList<Entry<E, Double>>(entrySet());
		Collections.sort(sorted, new EntryValueComparator(true));
		return sorted;
	}

	public Collection<Entry<E, Double>> getEntriesSortedByIncreasingCount() {
		final List<Entry<E, Double>> sorted = new ArrayList<Entry<E, Double>>(entrySet());
		Collections.sort(sorted, new EntryValueComparator(false));
		return sorted;
	}

	public Set<Map.Entry<E, Double>> getEntrySet() {
		return entries.entrySet();
	}

	/**
	 * I know, I know, this should be wrapped in a Distribution class, but it'text such a common use...why not. Returns the MLE prob.
	 * Assumes all the counts are >= 0.0 and totalCount > 0.0. If the latter is false, return 0.0 (i.e. 0/0 == 0)
	 * 
	 * @author Aria
	 * @param key
	 * @return MLE prob of the key
	 */
	public double getProbability(final E key) {
		final double count = getCount(key);
		final double total = totalCount();
		if (total < 0.0) {
			throw new RuntimeException("Can't call getProbability() with totalCount < 0.0");
		}

		double ret = 0;
		if (total > 0) {
			ret = count / total;
			if (ret > 1) {
				ret = 1;
			}
		}
		return ret;
	}

	public List<E> getSortedKeys() {
		return getSortedKeys(false);
	}

	public List<E> getSortedKeys(boolean ascending) {
		List<E> ret = new ArrayList<E>();

		if (ascending) {
			for (Entry<E, Double> entry : getEntriesSortedByIncreasingCount()) {
				ret.add(entry.getKey());
			}
		} else {
			for (Entry<E, Double> entry : getEntriesSortedByDecreasingCount()) {
				ret.add(entry.getKey());
			}
		}
		return ret;
	}

	/**
	 * Increment each element in a given collection by a given amount.
	 */
	public void incrementAll(final Collection<? extends E> collection, final double count) {
		for (final E key : collection) {
			incrementCount(key, count);
		}
		dirty = true;
	}

	public <T extends E> void incrementAll(final Counter<T> counter) {
		incrementAll(counter, 1.0);
	}

	public <T extends E> void incrementAll(final Counter<T> counter, final double scale) {
		for (final Entry<T, Double> entry : counter.entrySet()) {
			incrementCount(entry.getKey(), scale * entry.getValue());
		}
		dirty = true;
	}

	public void incrementAll(final double increment) {
		for (E key : entries.keySet()) {
			double newVal = getCount(key) + increment;
			setCount(key, newVal);
		}
		dirty = true;
	}

	public void incrementAll(final E[] items) {
		for (E item : items) {
			incrementCount(item, 1);
		}
		dirty = true;
	}

	/**
	 * Increment a key'text cnt by the given amount.
	 * 
	 * @param key
	 * @param increment
	 */
	public double incrementCount(final E key, final double increment) {
		final double newVal = getCount(key) + increment;
		setCount(key, newVal);
		dirty = true;
		return newVal;
	}

	/**
	 * True if there are no ents in the counter (false does not mean totalCount > 0)
	 */
	public boolean isEmpty() {
		return size() == 0;
	}

	public boolean isEqualTo(final Counter<E> counter) {
		boolean tmp = true;
		final Counter<E> bigger = counter.size() > size() ? counter : this;
		for (final E e : bigger.keySet()) {
			tmp &= counter.getCount(e) == getCount(e);
		}
		return tmp;
	}

	/**
	 * @author ohs
	 * 
	 * @param topN
	 */
	public void keepTopNKeys(int topN) {
		int cnt = 0;
		for (Entry<E, Double> entry : getEntriesSortedByDecreasingCount()) {
			if (++cnt > topN) {
				removeKey(entry.getKey());
			}
		}
	}

	/**
	 * The elements in the counter.
	 * 
	 * @return set of keys
	 */
	public Set<E> keySet() {
		return entries.keySet();
	}

	public double max() {
		return maxMinHelp(true);
	}

	private double maxMinHelp(final boolean max) {
		double maxCount = max ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;

		for (final Map.Entry<E, Double> entry : entries.entrySet()) {
			if ((max && entry.getValue() > maxCount) || (!max && entry.getValue() < maxCount)) {

				maxCount = entry.getValue();
			}
		}
		return maxCount;
	}

	public double min() {
		return maxMinHelp(false);
	}

	/**
	 * Destructively normalize this Counter in place.
	 */
	public void normalize() {
		final double totalCount = totalCount();
		for (final E key : keySet()) {
			setCount(key, getCount(key) / totalCount);
		}
		dirty = true;
	}

	public void prune(final Set<E> toRemove) {
		for (final E e : toRemove) {
			removeKey(e);
		}
	}

	public void pruneExcept(final Set<E> toKeep) {
		final List<E> toRemove = new ArrayList<E>();
		for (final E key : entries.keySet()) {
			if (!toKeep.contains(key))
				toRemove.add(key);
		}
		for (final E e : toRemove) {
			removeKey(e);
		}
	}

	public void pruneKeysBelowThreshold(final double cutoff) {
		final Iterator<E> it = entries.keySet().iterator();
		while (it.hasNext()) {
			final E key = it.next();
			final double val = entries.get(key);
			if (val < cutoff) {
				it.remove();
			}
		}
		dirty = true;
	}

	public void pruneKeysOverThreshold(final double cutoff) {
		final Iterator<E> it = entries.keySet().iterator();
		while (it.hasNext()) {
			final E key = it.next();
			final double val = entries.get(key);
			if (val > cutoff) {
				it.remove();
			}
		}
		dirty = true;
	}

	/**
	 * Set the cnt for the given key if it is larger than the previous one;
	 * 
	 * @param key
	 * @param cnt
	 */
	public void put(final E key, final double count, final boolean keepHigher) {
		if (keepHigher && entries.containsKey(key)) {
			final double oldCount = entries.get(key);
			if (count > oldCount) {
				entries.put(key, count);
			}
		} else {
			entries.put(key, count);
		}
		dirty = true;
	}

	public void putAll(final double d) {
		for (final Entry<E, Double> entry : entries.entrySet()) {

			setCount(entry.getKey(), d);
		}
		dirty = true;
	}

	/**
	 * @author ohs
	 * 
	 * @return
	 */
	public Counter<E> rank() {
		Counter<E> ret = new Counter<E>();
		double rank = 0;
		for (Entry<E, Double> entry : getEntriesSortedByDecreasingCount()) {
			ret.setCount(entry.getKey(), ++rank);
		}
		return ret;
	}

	public void removeKey(final E key) {
		setCount(key, 0.0);
		dirty = true;
		removeKeyFromEntries(key);
	}

	/**
	 * @param key
	 */
	protected void removeKeyFromEntries(final E key) {
		entries.remove(key);
	}

	public void replace(E key1, E key2) {
		Double value = entries.remove(key1);
		if (value != null) {
			incrementCount(key2, value);
		}
	}

	/**
	 * Will return a sample from the counter, will throw exception if any of the counts are < 0.0 or if the totalCount() <= 0.0
	 * 
	 * @return
	 * 
	 * @author aria42
	 */
	public E sample() {
		return sample(new Random());
	}

	/**
	 * Will return a sample from the counter, will throw exception if any of the counts are < 0.0 or if the totalCount() <= 0.0
	 * 
	 * @return
	 * 
	 * @author aria42
	 */
	public E sample(final Random rand) {
		final double total = totalCount();
		if (total <= 0.0) {
			throw new RuntimeException(String.format("Attempting to sample() with totalCount() %.3f\n", total));
		}
		double sum = 0.0;
		final double r = rand.nextDouble();
		for (final Map.Entry<E, Double> entry : entries.entrySet()) {
			final double count = entry.getValue();
			final double frac = count / total;
			sum += frac;
			if (r < sum) {
				return entry.getKey();
			}
		}
		throw new IllegalStateException("Shoudl've have returned a sample by now....");
	}

	public void scale(final double c) {
		for (final Map.Entry<E, Double> entry : getEntrySet()) {
			entry.setValue(entry.getValue() * c);
		}
	}

	public void scale(E key, final double c) {
		final double newVal = getCount(key) * c;
		setCount(key, newVal);
		dirty = true;
	}

	public Counter<E> scaledClone(final double c) {
		final Counter<E> newCounter = new Counter<E>();

		for (final Map.Entry<E, Double> entry : getEntrySet()) {
			newCounter.setCount(entry.getKey(), entry.getValue() * c);
		}

		return newCounter;
	}

	/**
	 * Sets all counts to the given value, but does not remove any keys
	 */
	public void setAllCounts(final double val) {
		for (final E e : keySet()) {
			setCount(e, val);
		}
	}

	/**
	 * Set the cnt for the given key, clobbering any previous cnt.
	 * 
	 * @param key
	 * @param cnt
	 */
	public void setCount(final E key, final double count) {
		entries.put(key, count);
		dirty = true;
	}

	public void setDefaultCount(final double deflt) {
		this.defaultCount = deflt;
	}

	public void setDirty(final boolean dirty) {
		this.dirty = dirty;
	}

	/**
	 * Set'text the key'text cnt to the maximum of the current cnt and val. Always sets to val if key is not yet present.
	 * 
	 * @param key
	 * @param val
	 */
	public void setMaxCount(final E key, final double val) {
		final Double value = entries.get(key);
		if (value == null || val > value.doubleValue()) {
			setCount(key, val);

			dirty = true;
		}
	}

	/**
	 * Set'text the key'text cnt to the minimum of the current cnt and val. Always sets to val if key is not yet present.
	 * 
	 * @param key
	 * @param val
	 */
	public void setMinCount(final E key, final double val) {
		final Double value = entries.get(key);
		if (value == null || val < value.doubleValue()) {
			setCount(key, val);

			dirty = true;
		}
	}

	/**
	 * The number of ents in the counter (not the total cnt -- use totalCount() instead).
	 */
	public int size() {
		return entries.size();
	}

	public Counter<E> toLogSpace() {
		final Counter<E> newCounter = new Counter<E>(this);
		for (final E key : newCounter.keySet()) {
			newCounter.setCount(key, Math.log(getCount(key)));
		}
		return newCounter;
	}

	/**
	 * Returns a string representation with the keys ordered by decreasing counts.
	 * 
	 * @return string representation
	 */

	@Override
	public String toString() {
		return toStringSortedByValues(true, false, 50, " ");
	}

	public String toString(int size) {
		return toStringSortedByValues(true, false, size, " ");
	}

	public String toStringSortedByKeys() {
		final StringBuilder sb = new StringBuilder("[");

		final NumberFormat f = NumberFormat.getInstance();
		f.setMaximumFractionDigits(5);
		int numKeysPrinted = 0;

		for (final E element : new TreeSet<E>(keySet())) {
			sb.append(element.toString());
			sb.append(" : ");
			sb.append(f.format(getCount(element)));
			if (numKeysPrinted < size() - 1)
				sb.append(", ");
			numKeysPrinted++;
		}
		if (numKeysPrinted < size())
			sb.append("...");
		sb.append("]");

		return sb.toString();
	}

	public String toStringSortedByValues(boolean descending, boolean vertical, int size, String delimeter) {
		final StringBuilder sb = new StringBuilder("[");
		if (vertical) {
			sb.append("\n");
		}

		int numKeys = 0;

		for (final Entry<E, Double> element : descending ? getEntriesSortedByDecreasingCount() : getEntriesSortedByIncreasingCount()) {
			double val = element.getValue();
			int tmp = (int) val;

			if (val - tmp == 0) {
				sb.append(String.format("%s:%d", element.getKey().toString(), tmp));
			} else {
				sb.append(String.format("%s:%f", element.getKey().toString(), val));
			}

			if (numKeys < size() - 1) {
				if (vertical) {
					sb.append("\n");
				} else {
					sb.append(delimeter);
				}
			}
			if (++numKeys == size) {
				break;
			}
		}

		if (size < size()) {
			sb.append("...");
		}
		if (vertical) {
			sb.append("\n");
		}
		sb.append("]");
		return sb.toString();
	}

	/**
	 * Finds the total of all counts in the counter. This implementation iterates through the entire counter every time this method is
	 * called.
	 * 
	 * @return the counter'text total
	 */
	public double totalCount() {
		if (!dirty) {
			return cacheTotal;
		}
		double total = 0.0;
		for (final Map.Entry<E, Double> entry : entries.entrySet()) {
			total += entry.getValue();
		}
		cacheTotal = total;
		dirty = false;
		return total;
	}

	public void trimToSize() {
		Map<E, Double> m = new HashMap<E, Double>(entries.size());
		m.putAll(entries);
		entries.clear();
		entries = null;
		entries = m;
	}

	public Iterable<Double> values() {
		return new Iterable<Double>() {

			@Override
			public Iterator<Double> iterator() {

				return new Iterator<Double>() {
					Iterator<Entry<E, Double>> entryIterator = entrySet().iterator();

					@Override
					public boolean hasNext() {
						return entryIterator.hasNext();
					}

					@Override
					public Double next() {
						return entryIterator.next().getValue();
					}

					@Override
					public void remove() {
						entryIterator.remove();
					}
				};
			}
		};
	}

}
