package ohs.math;

import java.util.Map.Entry;

import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.utils.Generics;

public class CounterMath {

	public static <E> void add(Counter<E> a, Counter<E> c, Counter<E> e) {
		add(a, 0, e);
		add(c, 0, e);
	}

	public static <E> void add(Counter<E> a, double b, Counter<E> c) {
		for (Entry<E, Double> e : a.entrySet()) {
			c.setCount(e.getKey(), b + e.getValue());
		}
	}

	public static <K, V> void add(CounterMap<K, V> a, CounterMap<K, V> b, CounterMap<K, V> c) {
		addAfterMultiply(a, 1, b, 1, c);
	}

	public static <K, V> void add(CounterMap<K, V> a, double b, CounterMap<K, V> c) {
		for (Entry<K, Counter<V>> e : a.getEntrySet()) {
			add(e.getValue(), b, c.getCounter(e.getKey()));
		}
	}

	public static <E> void addAfterMultiply(Counter<E> a, double ac, Counter<E> b) {
		for (Entry<E, Double> e : a.entrySet()) {
			E key = e.getKey();
			double val = e.getValue();
			double new_val = ac * val + b.getCount(key);
			b.setCount(key, new_val);
		}
	}

	public static <E> void addAfterMultiply(Counter<E> a, double ac, Counter<E> b, double bc, Counter<E> c) {
		addAfterMultiply(a, ac, c);
		addAfterMultiply(b, bc, c);
	}

	public static <E> Counter<E> addAfterMultiply(Counter<E> a, double ac, Counter<E> b, double bc) {
		Counter<E> c = Generics.newCounter();
		addAfterMultiply(a, ac, b, bc, c);
		return c;
	}

	public static <K, V> void addAfterMultiply(CounterMap<K, V> a, double ac, CounterMap<K, V> b) {
		for (Entry<K, Counter<V>> e : a.getEntrySet()) {
			addAfterMultiply(e.getValue(), ac, b.getCounter(e.getKey()));
		}
	}

	public static <K, V> void addAfterMultiply(CounterMap<K, V> a, double ac, CounterMap<K, V> b, double bc, CounterMap<K, V> c) {
		addAfterMultiply(a, ac, c);
		addAfterMultiply(b, bc, c);
	}

	public static <E> Counter<E> multiply(Counter<E> a, Counter<E> b) {
		Counter<E> c = Generics.newCounter();
		multiply(a, b, c);
		return c;
	}

	public static <E> void multiply(Counter<E> a, Counter<E> b, Counter<E> c) {
		Counter<E> small = a;
		Counter<E> large = b;

		if (small.size() > large.size()) {
			small = b;
			large = a;
		}

		for (Entry<E, Double> e : small.entrySet()) {
			E key = e.getKey();
			double val1 = e.getValue();
			double val2 = large.getCount(key);
			double val3 = val1 * val2;
			if (val3 != 0) {
				c.setCount(key, val3);
			}
		}
	}

	public static <K, V> void multiply(CounterMap<K, V> a, CounterMap<K, V> b, CounterMap<K, V> c) {
		CounterMap<K, V> small = a;
		CounterMap<K, V> large = b;

		if (small.size() > large.size()) {
			small = b;
			large = a;
		}

		for (K key : small.keySet()) {
			Counter<V> c1 = small.getCounter(key);
			Counter<V> c2 = large.getCounter(key);
			Counter<V> c3 = multiply(c1, c2);
			if (c3.size() > 0) {
				c.setCounter(key, c3);
			}
		}
	}

}
