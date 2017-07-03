package ohs.types.generic;

import java.io.Serializable;
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

	private Map<K, List<V>> ents;

	private ListType lt;

	private MapType mt;

	public ListMap() {
		this(0, Generics.MapType.HASH_MAP, Generics.ListType.ARRAY_LIST);
	}

	public ListMap(int size, MapType mt, ListType lt) {
		ents = Generics.newMap(mt, size);
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
			for (List<V> p : ents.values()) {
				p.clear();
			}
		}
		ents.clear();
	}

	public boolean containsKey(K key) {
		return ents.containsKey(key);
	}

	public List<V> ensure(K key) {
		List<V> ret = ents.get(key);
		if (ret == null) {
			ret = Generics.newList(lt);
			ents.put(key, ret);
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
		if (ents == null) {
			if (other.ents != null)
				return false;
		} else if (!ents.equals(other.ents))
			return false;
		return true;
	}

	public List<V> get(K key) {
		return get(key, true);
	}

	public List<V> get(K key, boolean ensure) {
		return ensure ? ensure(key) : ents.get(key);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((ents == null) ? 0 : ents.hashCode());
		return result;
	}

	public Set<K> keySet() {
		return ents.keySet();
	}

	public void put(K key, List<V> values) {
		ents.put(key, values);
	}

	public void put(K key, V value) {
		ensure(key).add(value);
	}

	public void put(ListMap<K, V> lm) {
		for (K k : lm.keySet()) {
			List<V> l = lm.get(k);
			get(k, true).addAll(l);
		}
	}

	public List<V> remove(K key) {
		return ents.remove(key);
	}

	public int size() {
		return ents.size();
	}

	public int sizeOfEntries() {
		int ret = 0;
		for (K key : ents.keySet()) {
			ret += ents.get(key).size();
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

		for (K key : ents.keySet()) {
			if (++cnt1 > row_size) {
				sb.append("\n...");
				break;
			}

			sb.append(key.toString() + " => ");
			sb.append("[ ");

			List<V> list = ents.get(key);

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
		Map<K, List<V>> temp = Generics.newMap(mt, ents.size());

		for (K k : ents.keySet()) {
			List<V> l = ents.get(k);
			List<V> nl = Generics.newArrayList(l.size());

			for (V v : l) {
				nl.add(v);
			}
			temp.put(k, nl);

			l.clear();
			l = null;
		}
		ents = temp;
	}

}
