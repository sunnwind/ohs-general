package ohs.types.generic;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;

import ohs.utils.Generics;
import ohs.utils.Generics.MapType;

public class MapMap<K, E, V> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7366754514015096846L;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	}

	protected Map<K, Map<E, V>> ents;

	private MapType mt1;

	protected MapType mt2;

	public MapMap() {
		this(100, MapType.HASH_MAP, MapType.HASH_MAP);
	}

	public MapMap(int size, MapType mt1, MapType mt2) {
		ents = Generics.newMap(mt1, size);
		this.mt1 = mt1;
		this.mt2 = mt2;
	}

	public void clear() {
		for (Map<E, V> m : ents.values()) {
			m.clear();
		}
		ents.clear();
	}

	public boolean containsKey(K key) {
		return ents.containsKey(key);
	}

	public boolean containsKeys(K key1, E key2) {
		Map<E, V> map = ents.get(key1);
		if (map == null) {
			return false;
		}
		if (!map.containsKey(key2)) {
			return false;
		}
		return true;
	}

	protected Map<E, V> ensure(K key) {
		Map<E, V> map = ents.get(key);
		if (map == null) {
			map = Generics.newMap(mt2);
			ents.put(key, map);
		}
		return map;
	}

	public Map<E, V> get(K key) {
		return get(key, true);
	}

	public Map<E, V> get(K key, boolean ensure) {
		return ensure ? ensure(key) : ents.get(key);
	}

	public V get(K key, E elem, boolean ensure) {
		Map<E, V> map = get(key, ensure);
		return map == null ? null : map.get(elem);
	}

	public Set<K> keySet() {
		return ents.keySet();
	}

	public Set<E> keySet(K key) {
		return ents.get(key).keySet();
	}

	public void put(K key, E elem, V value) {
		ensure(key).put(elem, value);
	}

	public void put(K key, Map<E, V> map) {
		ents.put(key, map);
	}

	public int size() {
		return ents.size();
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for (K key : ents.keySet()) {
			sb.append(key.toString() + " => ");
			Map<E, V> map = ents.get(key);
			int size = map.size();
			int num = 0;
			for (E elem : map.keySet()) {
				V value = map.get(elem);
				sb.append(elem.toString() + ":" + value.toString() + (++num >= size ? "" : " "));
			}
			sb.append("\n");
		}

		return sb.toString();
	}

	public long totalSize() {
		long ret = 0;
		for (Map<E, V> s : ents.values()) {
			ret += s.size();
		}
		return ret;
	}

	public void trimToSize() {
		Map<K, Map<E, V>> tmp = Generics.newMap(mt1, ents.size());
		for (K key : ents.keySet()) {
			Map<E, V> m = ents.get(key);
			Map<E, V> nm = Generics.newMap(mt2, m.size());
			nm.putAll(m);
			m.clear();

			tmp.put(key, nm);
		}
		ents = tmp;
	}

}
