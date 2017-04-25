package ohs.types.generic;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class Indexer<E> extends AbstractList<E> implements Serializable {
	private static final long serialVersionUID = -8769544079136550516L;
	protected List<E> objs;
	protected Map<E, Integer> idxs;

	public Indexer() {
		objs = new ArrayList<E>();
		idxs = new HashMap<E, Integer>();
	}

	public Indexer(Collection<? extends E> a) {
		this(a, false);
	}

	public Indexer(Collection<? extends E> a, boolean unique_input) {
		this();

		if (unique_input) {
			objs = new ArrayList<E>(a.size());
			idxs = new HashMap<E, Integer>(a.size());
			for (E b : a) {
				idxs.put(b, size());
				objs.add(b);
			}
		} else {
			for (E b : a) {
				add(b);
			}
		}
	}

	public Indexer(int size) {
		objs = new ArrayList<E>(size);
		idxs = new HashMap<E, Integer>(size);
	}

	@Override
	public boolean add(E elem) {
		if (contains(elem)) {
			return false;
		}
		idxs.put(elem, size());
		objs.add(elem);
		return true;
	}

	@Override
	public void clear() {
		objs.clear();
		idxs.clear();
	}

	@Override
	public boolean contains(Object o) {
		return idxs.keySet().contains(o);
	}

	public E get(int index) {
		return getObject(index);
	}

	public int getIndex(E e) {
		return getIndex(e, -1);
	}

	public int getIndex(E e, int unknown) {
		if (e == null)
			return unknown;
		Integer index = idxs.get(e);
		if (index == null) {
			index = size();
			objs.add(e);
			idxs.put(e, index);
		}
		return index;
	}

	public List<Integer> getIndexes(Collection<E> objs) {
		return getIndexes(objs, -1);
	}

	public List<Integer> getIndexes(Collection<E> objs, int unknown) {
		List<Integer> ret = new ArrayList<Integer>();
		for (E obj : objs) {
			ret.add(getIndex(obj, unknown));
		}
		return ret;
	}

	public IntegerArray getIndexes(E[] objs) {
		return getIndexes(objs, -1);
	}

	public IntegerArray getIndexes(E[] objs, int unknown) {
		IntegerArray ret = new IntegerArray(objs.length);
		for (int i = 0; i < objs.length; i++) {
			ret.add(getIndex(objs[i], unknown));
		}
		return ret;
	}

	public E getObject(int idx) {
		return objs.get(idx);
	}

	public List<E> getObjects() {
		return objs;
	}

	public List<E> getObjects(Collection<Integer> ids) {
		List<E> ret = new ArrayList<E>();
		for (int id : ids) {
			ret.add(getObject(id));
		}
		return ret;
	}

	public E[] getObjects(int[] idxs) {
		if (size() == 0)
			throw new IllegalArgumentException("bad");
		int n = idxs.length;
		Class c = objs.get(0).getClass();
		E[] os = (E[]) Array.newInstance(c, n);
		for (int i = 0; i < n; i++)
			os[i] = idxs[i] == -1 ? null : getObject(idxs[i]);
		return os;
	}

	public E[] getObjects(IntegerArray idxs) {
		if (size() == 0)
			throw new IllegalArgumentException("bad");
		int n = idxs.size();
		Class c = objs.get(0).getClass();
		E[] os = (E[]) Array.newInstance(c, n);
		for (int i = 0; i < n; i++)
			os[i] = idxs.get(i) == -1 ? null : getObject(idxs.get(i));

		return os;
	}

	public List<Integer> indexesOf(Collection<Object> objs, int unknown) {
		List<Integer> ret = new ArrayList<>(objs.size());
		for (Object obj : objs) {
			ret.add(indexOf(obj, unknown));
		}
		return ret;
	}

	public IntegerArray indexesOf(E[] objs) {
		return indexesOf(objs, -1);
	}

	public IntegerArray indexesOf(E[] objs, int unknown) {
		IntegerArray ret = new IntegerArray(objs.length);
		for (int i = 0; i < objs.length; i++) {
			ret.add(indexOf(objs[i], unknown));
		}
		return ret;
	}

	public List<Integer> indexesOfKnown(Collection<E> objs) {
		List<Integer> ret = new ArrayList<Integer>(objs.size());
		for (Object obj : objs) {
			int id = indexOf(obj);
			if (id > -1) {
				ret.add(id);
			}
		}
		return ret;
	}

	public IntegerArray indexesOfKnown(E[] objs) {
		IntegerArray ret = new IntegerArray(objs.length);
		for (Object obj : objs) {
			int id = indexOf(obj);
			if (id > -1) {
				ret.add(id);
			}
		}
		return ret;
	}

	@Override
	public int indexOf(Object obj) {
		return indexOf(obj, -1);
	}

	public int indexOf(Object o, int unknown) {
		Integer index = idxs.get(o);
		if (index == null)
			return unknown;
		return index;
	}

	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("size: %d", size()));
		return sb.toString();
	}

	@Override
	public int size() {
		return objs.size();
	}

	public String toString() {
		return objs.toString();
	}

	public void trimToSize() {
		((ArrayList<E>) objs).trimToSize();
		Map<E, Integer> newIdxs = Generics.newHashMap(objs.size());
		newIdxs.putAll(idxs);
		idxs = newIdxs;
	}

}
