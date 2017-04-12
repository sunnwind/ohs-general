package ohs.types.generic;

import java.io.Serializable;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.utils.Generics;
import ohs.utils.Generics.ListType;
import ohs.utils.Generics.MapType;

public class ListMap<K, V> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7366754514015096846L;

	private Map<K, List<V>> entries;

	private List<List<V>> ls;

	private ListType lt;

	private MapType mt;

	private int pointer = 0;

	public ListMap() {
		this(0, Generics.MapType.HASH_MAP, Generics.ListType.ARRAY_LIST);
	}

	public ListMap(int size, MapType mt, ListType lt) {
		entries = Generics.newMap(mt, size);
		ls = Generics.newArrayList(size);

		this.mt = mt;
		this.lt = lt;
	}

	public ListMap(ListType lt) {
		this(0, Generics.MapType.HASH_MAP, lt);
	}

	public void clear() {
		clear(true);
	}

	public void clear(boolean deep_clear) {
		if (deep_clear) {
			for (List<V> l : entries.values()) {
				l.clear();
			}
			ls.clear();
			entries.clear();
		} else {
			// Collections.sort(ls, new Comparator<List<V>>() {
			//
			// @Override
			// public int compare(List<V> o1, List<V> o2) {
			// return o1.size() < o2.size() ? 1 : -1;
			// }
			// });

			for (List<V> l : ls) {
				l.clear();
			}
			entries.clear();
		}

		pointer = 0;
	}

	public boolean containsKey(K key) {
		return entries.containsKey(key);
	}

	public List<V> ensure(int i) {
		if (ls.size() <= i) {
			int size = i - ls.size() + 1;
			for (int j = 0; j < size; j++) {
				ls.add(Generics.newList(lt));
			}
		}
		return ls.get(i);
	}

	public List<V> ensure(K key) {
		List<V> ret = entries.get(key);
		if (ret == null) {
			ret = ensure(pointer++);
			entries.put(key, ret);
		}
		return ret;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		ListMap other = (ListMap) obj;
		if (entries == null) {
			if (other.entries != null)
				return false;
		} else if (!entries.equals(other.entries))
			return false;
		return true;
	}

	public List<V> get(K key) {
		return get(key, true);
	}

	public List<V> get(K key, boolean ensure) {
		return ensure ? ensure(key) : entries.get(key);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((entries == null) ? 0 : entries.hashCode());
		return result;
	}

	public Set<K> keySet() {
		return entries.keySet();
	}

	public void put(K key, List<V> values) {
		entries.put(key, values);
	}

	public void put(K key, V value) {
		ensure(key).add(value);
	}

	public List<V> remove(K key) {
		return entries.remove(key);
	}

	public int size() {
		return entries.size();
	}

	public int sizeOfEntries() {
		int ret = 0;
		for (K key : entries.keySet()) {
			ret += entries.get(key).size();
		}
		return ret;
	}

	@Override
	public String toString() {
		return toString(50, 50);
	}

	public String toString(int row_size, int col_size) {
		StringBuffer sb = new StringBuffer();

		int cnt1 = 0;

		for (K key : entries.keySet()) {
			if (++cnt1 > row_size) {
				sb.append("\n...");
				break;
			}

			sb.append(key.toString() + " => ");
			sb.append("[ ");

			List<V> list = entries.get(key);

			for (int i = 0; i < list.size(); i++) {
				if (i > 0) {
					sb.append(", ");
				}

				if (i == col_size - 1) {
					sb.append("...");
					break;
				} else {
					sb.append(list.get(i).toString());
				}
			}

			sb.append(" ]");
			sb.append("\n");
		}
		return sb.toString();
	}

	public void trimToSize() {
		Map<K, List<V>> temp = Generics.newMap(mt, entries.size());

		for (K k : entries.keySet()) {
			List<V> l = entries.get(k);
			List<V> nl = Generics.newArrayList(l.size());

			for (V v : l) {
				nl.add(v);
			}
			temp.put(k, nl);

			l.clear();
			l = null;
		}
		entries = temp;
	}

}
