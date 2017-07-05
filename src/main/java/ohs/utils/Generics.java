package ohs.utils;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.WeakHashMap;
import java.util.concurrent.ConcurrentHashMap;

import ohs.types.generic.BidMap;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.CounterMapMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListList;
import ohs.types.generic.ListMap;
import ohs.types.generic.ListMapMap;
import ohs.types.generic.MapMap;
import ohs.types.generic.Pair;
import ohs.types.generic.SetMap;
import ohs.types.generic.Triple;

/**
 * from Stanford Core NLP
 * 
 * A collection of utilities to make dealing with Java generics less painful and verbose. For example, rather than declaring
 *
 * <pre>
 * {@code  Map<String,ClassicCounter<List<String>>> = new HashMap<String,ClassicCounter<List<String>>>()}
 * </pre>
 *
 * you just call <code>Generics.newHashMap()</code>:
 *
 * <pre>
 * {@code Map<String,ClassicCounter<List<String>>> = Generics.newHashMap()}
 * </pre>
 *
 * Java type-inference will almost always just <em>do the right thing</em> (every once in a while, the compiler will get confused before you
 * do, so you might still occasionally have to specify the appropriate types).
 *
 * This class is based on the examples in Brian Goetz'text article <a href="http://www.ibm.com/developerworks/library/j-jtp02216.html">Java
 * theory and practice: The pseudo-typedef antipattern</a>.
 *
 * @author Ilya Sherman
 */
public class Generics {

	public static enum ListType {
		ARRAY_LIST, LINKED_LIST;
	}

	public static enum MapType {
		HASH_MAP, TREE_MAP, WEAK_HASH_MAP, IDENTIY_HASH_MAP;
	}

	public static enum SetType {
		HASH_SET, TREE_SET, WEAK_HASH_SET, IDENTITY_HASH_SET;
	}

	@SuppressWarnings("unchecked")
	public static <T> T cast(Object o) {
		return (T) o;
	}

	public static <E> ArrayList<E> ensureArrayList(ArrayList<E> a) {
		return a == null ? newArrayList() : a;
	}

	public static <K, V> BidMap<K, V> ensureBidMap(BidMap<K, V> a) {
		return a == null ? newBidMap() : a;
	}

	public static <E> Counter<E> ensureCounter(Counter<E> a) {
		return a == null ? newCounter() : a;
	}

	public static <K, V> HashMap<K, V> ensureHashMap(Map<K, V> a) {
		return a == null ? newHashMap() : (HashMap<K, V>) a;
	}

	public static <E> Set<E> ensureHashSet(Set<E> a) {
		return a == null ? newHashSet() : a;
	}

	public static <K, V> TreeMap<K, V> ensureTreeMap(Map<K, V> a) {
		return a == null ? newTreeMap() : (TreeMap<K, V>) a;
	}

	/* Collections */
	public static <E> ArrayList<E> newArrayList() {
		return new ArrayList<E>();
	}

	public static <E> ArrayList<E> newArrayList(Collection<? extends E> a) {
		return new ArrayList<E>(a);
	}

	public static <E> ArrayList<E> newArrayList(E[] a) {
		ArrayList<E> ret = new ArrayList<E>(a.length);
		for (E e : a) {
			ret.add(e);
		}
		return ret;
	}

	public static <E> ArrayList<E> newArrayList(int size) {
		return new ArrayList<E>(size);
	}

	public static <K, V> BidMap<K, V> newBidMap() {
		return new BidMap<K, V>();
	}

	public static <K, V> BidMap<K, V> newBidMap(int size) {
		return new BidMap<K, V>(size);
	}

	public static <K, V> ConcurrentHashMap<K, V> newConcurrentHashMap() {
		return new ConcurrentHashMap<K, V>();
	}

	public static <K, V> ConcurrentHashMap<K, V> newConcurrentHashMap(int initialCapacity) {
		return new ConcurrentHashMap<K, V>(initialCapacity);
	}

	public static <K, V> ConcurrentHashMap<K, V> newConcurrentHashMap(int initialCapacity, float loadFactor, int concurrencyLevel) {
		return new ConcurrentHashMap<K, V>(initialCapacity, loadFactor, concurrencyLevel);
	}

	public static <E> Counter<E> newCounter() {
		return new Counter<E>();
	}

	public static <E> Counter<E> newCounter(Counter<? extends E> c) {
		return new Counter<E>(c);
	}

	public static <E> Counter<E> newCounter(E[] c) {
		return new Counter<E>(c);
	}

	public static <E> Counter<E> newCounter(int size) {
		return new Counter<E>(size);
	}

	public static <K, V> CounterMap<K, V> newCounterMap() {
		return new CounterMap<K, V>();
	}

	public static <K, V> CounterMap<K, V> newCounterMap(CounterMap<K, V> cm) {
		return new CounterMap<K, V>(cm);
	}

	public static <K, V> CounterMap<K, V> newCounterMap(int size) {
		return new CounterMap<K, V>(size);
	}

	public static <K, V, F> CounterMapMap<K, V, F> newCounterMapMap() {
		return new CounterMapMap<K, V, F>();
	}

	/* Maps */
	public static <K, V> HashMap<K, V> newHashMap() {
		return new HashMap<K, V>();
	}

	public static <K, V> HashMap<K, V> newHashMap(int initialCapacity) {
		return new HashMap<K, V>(initialCapacity);
	}

	public static <K, V> HashMap<K, V> newHashMap(Map<? extends K, ? extends V> m) {
		return new HashMap<K, V>(m);
	}

	public static <E> HashSet<E> newHashSet() {
		return new HashSet<E>();
	}

	public static <E> HashSet<E> newHashSet(Collection<? extends E> c) {
		return new HashSet<E>(c);
	}

	public static <E> HashSet<E> newHashSet(int initialCapacity) {
		return new HashSet<E>(initialCapacity);
	}

	public static <K, V> IdentityHashMap<K, V> newIdentityHashMap() {
		return new IdentityHashMap<K, V>();
	}

	public static <K, V> IdentityHashMap<K, V> newIdentityHashMap(int size) {
		return new IdentityHashMap<K, V>(size);
	}

	public static <K> Set<K> newIdentityHashSet() {
		return Collections.newSetFromMap(Generics.<K, Boolean> newIdentityHashMap());
	}

	public static <K> Set<K> newIdentityHashSet(int size) {
		return Collections.newSetFromMap(Generics.<K, Boolean> newIdentityHashMap(size));
	}

	public static <T> Indexer<T> newIndexer() {
		return new Indexer<T>();
	}

	public static <T> Indexer<T> newIndexer(int size) {
		return new Indexer<T>(size);
	}

	public static <T> Indexer<T> newIndexer(Collection<? extends T> a) {
		return new Indexer<T>(a);
	}

	public static <E> LinkedList<E> newLinkedList() {
		return new LinkedList<E>();
	}

	public static <E> LinkedList<E> newLinkedList(Collection<? extends E> c) {
		return new LinkedList<E>(c);
	}

	public static <E> List<E> newList(ListType t) {
		return newList(t, 0);
	}

	public static <E> List<E> newList(ListType t, int size) {
		List<E> ret = null;
		if (t == ListType.ARRAY_LIST) {
			ret = size > 0 ? newArrayList(size) : newArrayList();
		} else if (t == ListType.LINKED_LIST) {
			ret = newLinkedList();
		}
		return ret;
	}

	public static <V> ListList<V> newListList(int size) {
		return new ListList<V>(size, ListType.ARRAY_LIST, ListType.ARRAY_LIST);
	}

	public static <V> ListList<V> newListList() {
		return newListList(0);
	}

	public static <V> ListList<V> newListList(int size, ListType lt1, ListType lt2) {
		return new ListList<V>(size, lt1, lt2);
	}

	public static <K, V> ListMap<K, V> newListMap() {
		return new ListMap<K, V>();
	}

	public static <K, V> ListMap<K, V> newListMap(int size) {
		return new ListMap<K, V>(size, MapType.HASH_MAP, ListType.ARRAY_LIST);
	}

	public static <K, V> ListMap<K, V> newListMap(ListType lt) {
		return new ListMap<K, V>(0, MapType.HASH_MAP, lt);
	}

	public static <K, V> ListMap<K, V> newListMap(int size, MapType mt, ListType lt) {
		return new ListMap<K, V>(size, mt, lt);
	}

	public static <K, E, V> ListMapMap<K, E, V> newListMapMap() {
		return new ListMapMap<K, E, V>();
	}

	public static <K, E, V> ListMapMap<K, E, V> newListMapMap(int size, ListType lt) {
		return new ListMapMap<K, E, V>(size, lt);
	}

	public static <K, E, V> ListMapMap<K, E, V> newListMapMap(int size) {
		return newListMapMap(size, ListType.ARRAY_LIST);
	}

	public static <K, E, V> ListMapMap<K, E, V> newListMapMap(ListType lt) {
		return new ListMapMap<K, E, V>(0, lt);
	}

	public static <K, V> Map<K, V> newMap(MapType mt) {
		return newMap(mt, 0);
	}

	public static <K, V> Map<K, V> newMap(MapType mt, int size) {
		Map<K, V> ret = null;
		if (mt == MapType.HASH_MAP) {
			ret = size > 0 ? newHashMap(size) : newHashMap();
		} else if (mt == MapType.TREE_MAP) {
			ret = newTreeMap();
		} else if (mt == MapType.WEAK_HASH_MAP) {
			ret = size > 0 ? newWeakHashMap(size) : newWeakHashMap();
		} else if (mt == MapType.IDENTIY_HASH_MAP) {
			ret = size > 0 ? newIdentityHashMap(size) : newIdentityHashMap();
		}
		return ret;
	}

	public static <K, E, V> MapMap<K, E, V> newMapMap() {
		return new MapMap<K, E, V>();
	}

	public static <K, E, V> MapMap<K, E, V> newMapMap(int size) {
		return new MapMap<K, E, V>(size, MapType.HASH_MAP, MapType.HASH_MAP);
	}

	/* Other */
	public static <T1, T2> Pair<T1, T2> newPair(T1 first, T2 second) {
		return new Pair<T1, T2>(first, second);
	}

	public static <E> Set<E> newSet(SetType t) {
		return newSet(t, 0);
	}

	public static <E> Set<E> newSet(SetType t, int size) {
		Set<E> ret = null;
		if (t == SetType.HASH_SET) {
			ret = size > 0 ? newHashSet(size) : newHashSet();
		} else if (t == SetType.TREE_SET) {
			ret = newTreeSet();
		} else if (t == SetType.IDENTITY_HASH_SET) {
			ret = size > 0 ? newIdentityHashSet(size) : newIdentityHashSet();
		}
		return ret;
	}

	public static <K, V> SetMap<K, V> newSetMap() {
		return new SetMap<K, V>();
	}

	public static <K, V> SetMap<K, V> newSetMap(int size) {
		return new SetMap<K, V>(size, MapType.HASH_MAP, SetType.HASH_SET);
	}

	public static <E> Stack<E> newStack() {
		return new Stack<E>();
	}

	public static <K, V> TreeMap<K, V> newTreeMap() {
		return new TreeMap<K, V>();
	}

	public static <E> TreeSet<E> newTreeSet() {
		return new TreeSet<E>();
	}

	public static <E> TreeSet<E> newTreeSet(Collection<E> s) {
		return new TreeSet<E>(s);
	}

	public static <E> TreeSet<E> newTreeSet(Comparator<? super E> comparator) {
		return new TreeSet<E>(comparator);
	}

	public static <T1, T2, T3> Triple<T1, T2, T3> newTriple(T1 first, T2 second, T3 third) {
		return new Triple<T1, T2, T3>(first, second, third);
	}

	// public static <E> Index<E> newIndex() {
	// return new HashIndex<E>();
	// }

	public static <K, V> WeakHashMap<K, V> newWeakHashMap() {
		return new WeakHashMap<K, V>();
	}

	public static <K, V> WeakHashMap<K, V> newWeakHashMap(int size) {
		return new WeakHashMap<K, V>(size);
	}

	public static <T> WeakReference<T> newWeakReference(T referent) {
		return new WeakReference<T>(referent);
	}

	public static <K> void nullify(Collection<K> c) {
		Iterator<K> iter = c.iterator();
		while (iter.hasNext()) {
			iter.next();
			iter.remove();
		}
		c = null;
	}

	// public static <T> Interner<T> newInterner() {
	// return new Interner<T>();
	// }

	// public static <T> SynchronizedInterner<T>
	// newSynchronizedInterner(Interner<T> interner) {
	// return new SynchronizedInterner<T>(interner);
	// }
	//
	// public static <T> SynchronizedInterner<T>
	// newSynchronizedInterner(Interner<T> interner, Object mutex) {
	// return new SynchronizedInterner<T>(interner, mutex);
	// }

	private Generics() {
	} // static class
}
