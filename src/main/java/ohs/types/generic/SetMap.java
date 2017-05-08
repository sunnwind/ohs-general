package ohs.types.generic;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;

import ohs.utils.Generics;

public class SetMap<K, V> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7366754514015096846L;

	protected Map<K, Set<V>> entries;

	private Generics.MapType mt;

	private Generics.SetType st;

	public SetMap() {
		this(100, Generics.MapType.HASH_MAP, Generics.SetType.HASH_SET);
	}

	public SetMap(Generics.MapType mt, Generics.SetType st) {
		this(100, mt, st);
	}

	public SetMap(int size, Generics.MapType mt, Generics.SetType st) {
		this.mt = mt;
		this.st = st;
		entries = Generics.newMap(mt, size);
	}

	public void addAll(SetMap<K, V> input) {
		for (K key : input.keySet()) {
			Set<V> set = ensure(key);
			for (V val : input.get(key, false)) {
				set.add(val);
			}
		}
	}

	public void clear() {
		for (K key : entries.keySet()) {
			Set<V> set = entries.get(key);
			set.clear();
		}
		entries.clear();
	}

	public boolean contains(K key, V value) {
		boolean ret = false;
		Set<V> set = entries.get(key);
		if (set != null && set.contains(value)) {
			ret = true;
		}
		return ret;
	}

	public boolean containsKey(K key) {
		return entries.containsKey(key);
	}

	protected Set<V> ensure(K key) {
		Set<V> set = entries.get(key);
		if (set == null) {
			set = Generics.newSet(st);
			entries.put(key, set);
		}
		return set;
	}

	public Set<V> get(K key) {
		return get(key, true);
	}

	public Set<V> get(K key, boolean ensure) {
		return ensure ? ensure(key) : entries.get(key);
	}

	public Set<Map.Entry<K, Set<V>>> getEntrySet() {
		return entries.entrySet();
	}

	public SetMap<V, K> invert() {
		SetMap<V, K> ret = Generics.newSetMap();
		for (K k : entries.keySet()) {
			for (V v : entries.get(k)) {
				ret.put(v, k);
			}
		}
		return ret;
	}

	public Set<K> keySet() {
		return entries.keySet();
	}

	public void put(K key, Set<V> values) {
		entries.put(key, values);
	}

	public void put(K key, V value) {
		ensure(key).add(value);
	}

	public Set<V> removeKey(K key) {
		return entries.remove(key);
	}

	public void replaceAll(V value1, V value2) {
		for (Set<V> set : entries.values()) {
			if (set.remove(value1)) {
				set.add(value2);
			}
		}
	}

	public void replace(K key, V value1, V value2) {
		Set<V> set = entries.get(key);
		if (set != null) {
			if (set.remove(value1)) {
				set.add(value2);
			}
		}
	}

	public int size() {
		return entries.size();
	}

	@Override
	public String toString() {
		return toString(100, 20);
	}

	public String toString(int num_print_keys, int num_print_values) {
		StringBuffer sb = new StringBuffer();
		int numKeys = 0;
		for (K key : entries.keySet()) {
			if (++numKeys > num_print_keys) {
				break;
			}

			sb.append(key.toString() + " => ");
			Set<V> set = entries.get(key);
			int size = set.size();
			int numValues = 0;
			for (V value : set) {
				sb.append(value.toString() + (++numValues >= size ? "" : ", "));
				if (numValues > num_print_keys) {
					sb.append("...");
					break;
				}
			}
			sb.append("\n");
		}
		return sb.toString();
	}

	public long totalSize() {
		long ret = 0;
		for (Set<V> s : entries.values()) {
			ret += s.size();
		}
		return ret;
	}

	public void trimToSize() {
		Map<K, Set<V>> temp = Generics.newMap(mt, entries.size());

		for (K key : entries.keySet()) {
			Set<V> oldSet = entries.get(key);
			Set<V> newSet = Generics.newSet(st, oldSet.size());
			newSet.addAll(oldSet);
			temp.put(key, newSet);
		}

		entries = temp;
	}

}
