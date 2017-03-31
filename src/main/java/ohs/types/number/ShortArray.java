package ohs.types.number;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.Collections;
import java.util.ConcurrentModificationException;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.RandomAccess;
import java.util.Spliterator;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;

import ohs.io.FileUtils;
import ohs.math.ArrayUtils;
import ohs.utils.ByteSize;

public class ShortArray implements RandomAccess, Cloneable, java.io.Serializable, Iterable<Short> {
	/** Index-based split-by-two, lazily initialized Spliterator */
	static final class ArrayListSpliterator<Short> implements Spliterator<Short> {

		/*
		 * If ArrayLists were immutable, or structurally immutable (no adds, removes, etc), we could implement their spliterators with
		 * Arrays.spliterator. Instead we detect as much interference during traversal as practical without sacrificing much performance. We
		 * rely primarily on modCounts. These are not guaranteed to detect concurrency violations, and are sometimes overly conservative
		 * about within-thread interference, but detect enough problems to be worthwhile in practice. To carry this out, we (1) lazily
		 * initialize fence and expectedModCount until the latest point that we need to commit to the state we are checking against; thus
		 * improving precision. (This doesn't apply to SubLists, that create spliterators with current non-lazy values). (2) We perform only
		 * a single ConcurrentModificationException check at the end of forEach (the most performance-sensitive method). When using forEach
		 * (as opposed to iterators), we can normally only detect interference after actions, not before. Further CME-triggering checks
		 * apply to all other possible violations of assumptions for example null or too-small vals array given its size(), that could only
		 * have occurred due to interference. This allows the inner loop of forEach to run without any further checks, and simplifies
		 * lambda-resolution. While this does entail a number of checks, note that in the common case of list.stream().forEach(a), no checks
		 * or other computation occur anywhere other than inside forEach itself. The other less-often-used methods cannot take advantage of
		 * most of these streamlinings.
		 */

		private final ShortArray list;
		private int index; // current index, modified on advance/split
		private int fence; // -1 until used; then one past last index
		private int expectedModCount; // initialized when fence set

		/** Create new spliterator covering the given range */
		ArrayListSpliterator(ShortArray list, int origin, int fence, int expectedModCount) {
			this.list = list; // OK if null unless traversed
			this.index = origin;
			this.fence = fence;
			this.expectedModCount = expectedModCount;
		}

		public int characteristics() {
			return Spliterator.ORDERED | Spliterator.SIZED | Spliterator.SUBSIZED;
		}

		public long estimateSize() {
			return (long) (getFence() - index);
		}

		public void forEachRemaining(Consumer action) {
			int i, hi, mc; // hoist accesses and checks from loop
			ShortArray lst;
			short[] a;
			if (action == null)
				throw new NullPointerException();
			if ((lst = list) != null && (a = lst.vals) != null) {
				if ((hi = fence) < 0) {
					mc = lst.modCount;
					hi = lst.size;
				} else
					mc = expectedModCount;
				if ((i = index) >= 0 && (index = hi) <= a.length) {
					for (; i < hi; ++i) {
						@SuppressWarnings("unchecked")
						int e = a[i];
						action.accept(e);
					}
					if (lst.modCount == mc)
						return;
				}
			}
			throw new ConcurrentModificationException();
		}

		private int getFence() { // initialize fence to size on first use
			int hi; // (a specialized variant appears in method forEach)
			ShortArray lst;
			if ((hi = fence) < 0) {
				if ((lst = list) == null)
					hi = fence = 0;
				else {
					expectedModCount = lst.modCount;
					hi = fence = lst.size;
				}
			}
			return hi;
		}

		public boolean tryAdvance(Consumer action) {
			if (action == null)
				throw new NullPointerException();
			int hi = getFence(), i = index;
			if (i < hi) {
				index = i + 1;
				@SuppressWarnings("unchecked")
				int e = list.vals[i];
				action.accept(e);
				if (list.modCount != expectedModCount)
					throw new ConcurrentModificationException();
				return true;
			}
			return false;
		}

		public ArrayListSpliterator<Short> trySplit() {
			int hi = getFence(), lo = index, mid = (lo + hi) >>> 1;
			return (lo >= mid) ? null : // divide range in half unless too small
					new ArrayListSpliterator<Short>(list, lo, index = mid, expectedModCount);
		}
	}

	/**
	 * An optimized version of AbstractList.Itr
	 */
	private class Itr implements Iterator<Short> {
		int cursor; // index of next element to return
		int lastRet = -1; // index of last element returned; -1 if no such
		int expectedModCount = modCount;

		final void checkForComodification() {
			if (modCount != expectedModCount)
				throw new ConcurrentModificationException();
		}

		@Override
		@SuppressWarnings("unchecked")
		public void forEachRemaining(Consumer consumer) {
			Objects.requireNonNull(consumer);
			final int size = ShortArray.this.size;
			int i = cursor;
			if (i >= size) {
				return;
			}
			final short[] elementData = ShortArray.this.vals;
			if (i >= elementData.length) {
				throw new ConcurrentModificationException();
			}
			while (i != size && modCount == expectedModCount) {
				consumer.accept(elementData[i++]);
			}
			// update once at end of iteration to reduce heap write traffic
			cursor = i;
			lastRet = i - 1;
			checkForComodification();
		}

		public boolean hasNext() {
			return cursor != size;
		}

		public Short next() {
			checkForComodification();
			int i = cursor;
			if (i >= size)
				throw new NoSuchElementException();
			short[] elementData = ShortArray.this.vals;
			if (i >= elementData.length)
				throw new ConcurrentModificationException();
			cursor = i + 1;
			return elementData[lastRet = i];
		}

		public void remove() {
			if (lastRet < 0)
				throw new IllegalStateException();
			checkForComodification();

			try {
				ShortArray.this.remove(lastRet);
				cursor = lastRet;
				lastRet = -1;
				expectedModCount = modCount;
			} catch (IndexOutOfBoundsException ex) {
				throw new ConcurrentModificationException();
			}
		}
	}

	/**
	 * An optimized version of AbstractList.ListItr
	 */
	private class ListItr extends Itr implements ListIterator<Short> {
		ListItr(int index) {
			super();
			cursor = index;
		}

		public void add(Short e) {
			checkForComodification();

			try {
				int i = cursor;
				ShortArray.this.add(i, e);
				cursor = i + 1;
				lastRet = -1;
				expectedModCount = modCount;
			} catch (IndexOutOfBoundsException ex) {
				throw new ConcurrentModificationException();
			}
		}

		public boolean hasPrevious() {
			return cursor != 0;
		}

		public int nextIndex() {
			return cursor;
		}

		@SuppressWarnings("unchecked")
		public Short previous() {
			checkForComodification();
			int i = cursor - 1;
			if (i < 0)
				throw new NoSuchElementException();
			short[] elementData = ShortArray.this.vals;
			if (i >= elementData.length)
				throw new ConcurrentModificationException();
			cursor = i;
			return elementData[lastRet = i];
		}

		public int previousIndex() {
			return cursor - 1;
		}

		public void set(Short e) {
			if (lastRet < 0)
				throw new IllegalStateException();
			checkForComodification();

			try {
				ShortArray.this.set(lastRet, e);
			} catch (IndexOutOfBoundsException ex) {
				throw new ConcurrentModificationException();
			}
		}
	}

	private static final long serialVersionUID = 8683452581122892189L;

	/**
	 * Default initial capacity.
	 */
	private static final int DEFAULT_CAPACITY = 10;

	/**
	 * Shared empty array instance used for empty instances.
	 */
	private static final short[] EMPTY_VALUES = {};

	/**
	 * Shared empty array instance used for default sized empty instances. We distinguish this from EMPTY_VALUES to know how much to inflate
	 * when first element is added.
	 */
	private static final short[] DEFAULT_CAPACITY_EMPTY_VALUES = {};

	/**
	 * The maximum size of array to allocate. Some VMs reserve some header words in an array. Attempts to allocate larger arrays may result
	 * in OutOfMemoryError: Requested array size exceeds VM limit
	 */
	private static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;

	private static int hugeCapacity(int minCapacity) {
		if (minCapacity < 0) // overflow
			throw new OutOfMemoryError();
		return (minCapacity > MAX_ARRAY_SIZE) ? Short.MAX_VALUE : MAX_ARRAY_SIZE;
	}

	static void subListRangeCheck(int fromIndex, int toIndex, int size) {
		if (fromIndex < 0)
			throw new IndexOutOfBoundsException("fromIndex = " + fromIndex);
		if (toIndex > size)
			throw new IndexOutOfBoundsException("toIndex = " + toIndex);
		if (fromIndex > toIndex)
			throw new IllegalArgumentException("fromIndex(" + fromIndex + ") > toIndex(" + toIndex + ")");
	}

	/**
	 * The array buffer into which the elements of the ArrayList are stored. The capacity of the ArrayList is the length of this array
	 * buffer. Any empty ArrayList with vals == DEFAULT_CAPACITY_EMPTY_VALUES will be expanded to DEFAULT_CAPACITY when the first element is
	 * added.
	 */
	transient short[] vals; // non-private to simplify nested class access

	/**
	 * The size of the ArrayList (the number of elements it contains).
	 *
	 * @serial
	 */
	private int size;

	private int modCount = 0;

	/**
	 * Constructs an empty list with an initial capacity of ten.
	 */
	public ShortArray() {
		this.vals = DEFAULT_CAPACITY_EMPTY_VALUES;
	}

	/**
	 * Constructs a list containing the elements of the specified collection, in the order they are returned by the collection's iterator.
	 *
	 * @param c
	 *            the collection whose elements are to be placed into this list
	 * @throws NullPointerException
	 *             if the specified collection is null
	 */
	public ShortArray(Collection<Short> c) {
		vals = ArrayUtils.copyShorts(c);

		size = vals.length;
	}

	/**
	 * Constructs an empty list with the specified initial capacity.
	 *
	 * @param initialCapacity
	 *            the initial capacity of the list
	 * @throws IllegalArgumentException
	 *             if the specified initial capacity is negative
	 */
	public ShortArray(int initialCapacity) {
		if (initialCapacity > 0) {
			this.vals = new short[initialCapacity];
		} else if (initialCapacity == 0) {
			this.vals = EMPTY_VALUES;
		} else {
			throw new IllegalArgumentException("Illegal Capacity: " + initialCapacity);
		}

	}

	public ShortArray(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public ShortArray(short[] elementData) {
		this.vals = elementData;
		size = elementData.length;
	}

	public ShortArray(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	/**
	 * Inserts the specified element at the specified position in this list. Shifts the element currently at that position (if any) and any
	 * subsequent elements to the right (adds one to their indices).
	 *
	 * @param index
	 *            index at which the specified element is to be inserted
	 * @param element
	 *            element to be inserted
	 * @throws IndexOutOfBoundsException
	 *             {@inheritDoc}
	 */
	public void add(int index, short element) {
		rangeCheckForAdd(index);

		ensureCapacityInternal(size + 1); // Increments modCount!!
		System.arraycopy(vals, index, vals, index + 1, size - index);
		vals[index] = element;
		size++;
	}

	/**
	 * Appends the specified element to the end of this list.
	 *
	 * @param e
	 *            element to be appended to this list
	 * @return <tt>true</tt> (as specified by {@link Collection#add})
	 */
	public boolean add(short e) {
		ensureCapacityInternal(size + 1); // Increments modCount!!
		vals[size++] = e;
		return true;
	}

	/**
	 * Appends all of the elements in the specified collection to the end of this list, in the order that they are returned by the specified
	 * collection's Iterator. The behavior of this operation is undefined if the specified collection is modified while the operation is in
	 * progress. (This implies that the behavior of this call is undefined if the specified collection is this list, and this list is
	 * nonempty.)
	 *
	 * @param c
	 *            collection containing elements to be added to this list
	 * @return <tt>true</tt> if this list changed as a result of the call
	 * @throws NullPointerException
	 *             if the specified collection is null
	 */
	public boolean addAll(Collection<Short> c) {
		int numNew = c.size();
		ensureCapacityInternal(size + numNew); // Increments modCount
		for (short v : c) {
			vals[size++] = v;
		}
		return numNew != 0;
	}

	/**
	 * Inserts all of the elements in the specified collection into this list, starting at the specified position. Shifts the element
	 * currently at that position (if any) and any subsequent elements to the right (increases their indices). The new elements will appear
	 * in the list in the order that they are returned by the specified collection's iterator.
	 *
	 * @param index
	 *            index at which to insert the first element from the specified collection
	 * @param c
	 *            collection containing elements to be added to this list
	 * @return <tt>true</tt> if this list changed as a result of the call
	 * @throws IndexOutOfBoundsException
	 *             {@inheritDoc}
	 * @throws NullPointerException
	 *             if the specified collection is null
	 */
	public boolean addAll(int index, Collection<Short> c) {
		rangeCheckForAdd(index);

		int numNew = c.size();
		ensureCapacityInternal(size + numNew); // Increments modCount

		int numMoved = size - index;
		if (numMoved > 0)
			System.arraycopy(vals, index, vals, index + numNew, numMoved);

		for (short v : c) {
			vals[index++] = v;
		}
		size += numNew;
		return numNew != 0;
	}

	public boolean addAll(ShortArray c) {
		int numNew = c.size();
		ensureCapacityInternal(size + numNew); // Increments modCount
		for (short v : c) {
			vals[size++] = v;
		}
		return numNew != 0;
	}

	private boolean batchRemove(Collection<Short> c, boolean complement) {
		final short[] elementData = this.vals;
		int r = 0, w = 0;
		boolean modified = false;
		try {
			for (; r < size; r++)
				if (c.contains(elementData[r]) == complement)
					elementData[w++] = elementData[r];
		} finally {
			// Preserve behavioral compatibility with AbstractCollection,
			// even if c.contains() throws.
			if (r != size) {
				System.arraycopy(elementData, r, elementData, w, size - r);
				w += size - r;
			}
			if (w != size) {
				// clear to let GC do its work
				for (int i = w; i < size; i++)
					elementData[i] = 0;
				modCount += size - w;
				size = w;
				modified = true;
			}
		}
		return modified;
	}

	public ByteSize byteSize() {
		return new ByteSize(Short.BYTES * size());
	}

	/**
	 * Removes all of the elements from this list. The list will be empty after this call returns.
	 */
	public void clear() {
		modCount++;

		// clear to let GC do its work
		for (int i = 0; i < size; i++)
			vals[i] = 0;

		size = 0;
	}

	/**
	 * Returns a shallow copy of this <tt>ArrayList</tt> instance. (The elements themselves are not copied.)
	 *
	 * @return a clone of this <tt>ArrayList</tt> instance
	 */
	public ShortArray clone() {
		try {
			ShortArray v = (ShortArray) super.clone();
			v.vals = Arrays.copyOf(vals, size);
			v.modCount = 0;
			return v;
		} catch (CloneNotSupportedException e) {
			// this shouldn't happen, since we are Cloneable
			throw new InternalError(e);
		}
	}

	/**
	 * Returns <tt>true</tt> if this list contains the specified element. More formally, returns <tt>true</tt> if and only if this list
	 * contains at least one element <tt>e</tt> such that <tt>(o==null&nbsp;?&nbsp;e==null&nbsp;:&nbsp;o.equals(e))</tt>.
	 *
	 * @param o
	 *            element whose presence in this list is to be tested
	 * @return <tt>true</tt> if this list contains the specified element
	 */
	public boolean contains(short o) {
		return indexOf(o) >= 0;
	}

	/**
	 * Increases the capacity of this <tt>ArrayList</tt> instance, if necessary, to ensure that it can hold at least the number of elements
	 * specified by the minimum capacity argument.
	 *
	 * @param minCapacity
	 *            the desired minimum capacity
	 */
	public void ensureCapacity(int minCapacity) {
		int minExpand = (vals != DEFAULT_CAPACITY_EMPTY_VALUES)
				// any size if not default element table
				? 0
				// larger than default for default empty table. It's already
				// supposed to be at default size.
				: DEFAULT_CAPACITY;

		if (minCapacity > minExpand) {
			ensureExplicitCapacity(minCapacity);
		}
	}

	private void ensureCapacityInternal(int minCapacity) {
		if (vals == DEFAULT_CAPACITY_EMPTY_VALUES) {
			minCapacity = Math.max(DEFAULT_CAPACITY, minCapacity);
		}

		ensureExplicitCapacity(minCapacity);
	}

	private void ensureExplicitCapacity(int minCapacity) {
		modCount++;

		// overflow-conscious code
		if (minCapacity - vals.length > 0)
			grow(minCapacity);
	}

	/*
	 * Private remove method that skips bounds checking and does not return the value removed.
	 */
	private void fastRemove(int index) {
		modCount++;
		int numMoved = size - index - 1;
		if (numMoved > 0)
			System.arraycopy(vals, index + 1, vals, index, numMoved);
		vals[--size] = 0; // clear to let GC do its work
	}

	public void forEach(Consumer action) {
		Objects.requireNonNull(action);
		final int expectedModCount = modCount;
		@SuppressWarnings("unchecked")
		final short[] elementData = this.vals;
		final int size = this.size;
		for (int i = 0; modCount == expectedModCount && i < size; i++) {
			action.accept(elementData[i]);
		}
		if (modCount != expectedModCount) {
			throw new ConcurrentModificationException();
		}
	}

	/**
	 * Returns the element at the specified position in this list.
	 *
	 * @param index
	 *            index of the element to return
	 * @return the element at the specified position in this list
	 * @throws IndexOutOfBoundsException
	 *             {@inheritDoc}
	 */
	public short get(int index) {
		rangeCheck(index);
		return value(index);
	}

	public ShortArray get(int[] locs) {
		ShortArray ret = new ShortArray(locs.length);
		for (int loc : locs) {
			ret.add(get(loc));
		}
		return ret;
	}

	// Positional Access Operations

	/**
	 * Increases the capacity to ensure that it can hold at least the number of elements specified by the minimum capacity argument.
	 *
	 * @param minCapacity
	 *            the desired minimum capacity
	 */
	private void grow(int minCapacity) {
		// overflow-conscious code
		int oldCapacity = vals.length;
		int newCapacity = oldCapacity + (oldCapacity >> 1);
		if (newCapacity - minCapacity < 0)
			newCapacity = minCapacity;
		if (newCapacity - MAX_ARRAY_SIZE > 0)
			newCapacity = hugeCapacity(minCapacity);
		// minCapacity is usually close to size, so this is a win:
		vals = Arrays.copyOf(vals, newCapacity);
	}

	/**
	 * Returns the index of the first occurrence of the specified element in this list, or -1 if this list does not contain the element.
	 * More formally, returns the lowest index <tt>i</tt> such that
	 * <tt>(o==null&nbsp;?&nbsp;get(i)==null&nbsp;:&nbsp;o.equals(get(i)))</tt>, or -1 if there is no such index.
	 */
	public int indexOf(short o) {
		for (int i = 0; i < size; i++)
			if (o == vals[i])
				return i;
		return -1;
	}

	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[Info]\n");
		sb.append(String.format("size:\t%d\n", size()));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	/**
	 * Returns <tt>true</tt> if this list contains no elements.
	 *
	 * @return <tt>true</tt> if this list contains no elements
	 */
	public boolean isEmpty() {
		return size == 0;
	}

	/**
	 * Returns an iterator over the elements in this list in proper sequence.
	 *
	 * <p>
	 * The returned iterator is <a href="#fail-fast"><i>fail-fast</i></a>.
	 *
	 * @return an iterator over the elements in this list in proper sequence
	 */
	public Iterator<Short> iterator() {
		return new Itr();
	}

	/**
	 * Returns the index of the last occurrence of the specified element in this list, or -1 if this list does not contain the element. More
	 * formally, returns the highest index <tt>i</tt> such that <tt>(o==null&nbsp;?&nbsp;get(i)==null&nbsp;:&nbsp;o.equals(get(i)))</tt>, or
	 * -1 if there is no such index.
	 */
	public int lastIndexOf(int o) {
		for (int i = size - 1; i >= 0; i--)
			if (o == vals[i])
				return i;
		return -1;
	}

	/**
	 * Returns a list iterator over the elements in this list (in proper sequence).
	 *
	 * <p>
	 * The returned list iterator is <a href="#fail-fast"><i>fail-fast</i></a>.
	 *
	 * @see #listIterator(int)
	 */
	public ListIterator<Short> listIterator() {
		return new ListItr(0);
	}

	/**
	 * Returns a list iterator over the elements in this list (in proper sequence), starting at the specified position in the list. The
	 * specified index indicates the first element that would be returned by an initial call to {@link ListIterator#next next}. An initial
	 * call to {@link ListIterator#previous previous} would return the element with the specified index minus one.
	 *
	 * <p>
	 * The returned list iterator is <a href="#fail-fast"><i>fail-fast</i></a>.
	 *
	 * @throws IndexOutOfBoundsException
	 *             {@inheritDoc}
	 */
	public ListIterator<Short> listIterator(int index) {
		if (index < 0 || index > size)
			throw new IndexOutOfBoundsException("Index: " + index);
		return new ListItr(index);
	}

	/**
	 * Constructs an IndexOutOfBoundsException detail message. Of the many possible refactorings of the error handling code, this
	 * "outlining" performs best with both server and client VMs.
	 */
	private String outOfBoundsMsg(int index) {
		return "Index: " + index + ", Size: " + size;
	}

	/**
	 * Checks if the given index is in range. If not, throws an appropriate runtime exception. This method does *not* check if the index is
	 * negative: It is always used immediately prior to an array access, which throws an ArrayIndexOutOfBoundsException if index is
	 * negative.
	 */
	private void rangeCheck(int index) {
		if (index >= size)
			throw new IndexOutOfBoundsException(outOfBoundsMsg(index));
	}

	/**
	 * A version of rangeCheck used by add and addAll.
	 */
	private void rangeCheckForAdd(int index) {
		if (index > size || index < 0)
			throw new IndexOutOfBoundsException(outOfBoundsMsg(index));
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		vals = DEFAULT_CAPACITY_EMPTY_VALUES;

		int size = ois.readInt();
		ensureCapacity(size);
		for (int i = 0; i < size; i++) {
			add(ois.readShort());
		}
	}

	public void readObject(String fileName) throws Exception {
		System.out.printf("read at [%s]\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	/**
	 * Removes the first occurrence of the specified element from this list, if it is present. If the list does not contain the element, it
	 * is unchanged. More formally, removes the element with the lowest index <tt>i</tt> such that
	 * <tt>(o==null&nbsp;?&nbsp;get(i)==null&nbsp;:&nbsp;o.equals(get(i)))</tt> (if such an element exists). Returns <tt>true</tt> if this
	 * list contained the specified element (or equivalently, if this list changed as a result of the call).
	 *
	 * @param o
	 *            element to be removed from this list, if present
	 * @return <tt>true</tt> if this list contained the specified element
	 */
	public boolean remove(int o) {
		for (int index = 0; index < size; index++) {
			if (o == vals[index]) {
				fastRemove(index);
				return true;
			}
		}
		return false;
	}

	/**
	 * Removes from this list all of its elements that are contained in the specified collection.
	 *
	 * @param c
	 *            collection containing elements to be removed from this list
	 * @return {@code true} if this list changed as a result of the call
	 * @throws ClassCastException
	 *             if the class of an element of this list is incompatible with the specified collection
	 *             (<a href="Collection.html#optional-restrictions">optional</a>)
	 * @throws NullPointerException
	 *             if this list contains a null element and the specified collection does not permit null elements
	 *             (<a href="Collection.html#optional-restrictions">optional</a>), or if the specified collection is null
	 * @see Collection#contains(Object)
	 */
	public boolean removeAll(Collection<Short> c) {
		Objects.requireNonNull(c);
		return batchRemove(c, false);
	}

	/**
	 * Removes the element at the specified position in this list. Shifts any subsequent elements to the left (subtracts one from their
	 * indices).
	 *
	 * @param index
	 *            the index of the element to be removed
	 * @return the element that was removed from the list
	 * @throws IndexOutOfBoundsException
	 *             {@inheritDoc}
	 */
	public int removeAt(int index) {
		rangeCheck(index);

		modCount++;
		int oldValue = value(index);

		int numMoved = size - index - 1;
		if (numMoved > 0)
			System.arraycopy(vals, index + 1, vals, index, numMoved);
		vals[--size] = 0; // clear to let GC do its work

		return oldValue;
	}

	public boolean removeIf(Predicate<Short> filter) {
		Objects.requireNonNull(filter);
		// figure out which elements are to be removed
		// any exception thrown from the filter predicate at this stage
		// will leave the collection unmodified
		int removeCount = 0;
		final BitSet removeSet = new BitSet(size);
		final int expectedModCount = modCount;
		final int size = this.size;
		for (int i = 0; modCount == expectedModCount && i < size; i++) {
			@SuppressWarnings("unchecked")
			final short element = vals[i];
			if (filter.test(element)) {
				removeSet.set(i);
				removeCount++;
			}
		}
		if (modCount != expectedModCount) {
			throw new ConcurrentModificationException();
		}

		// shift surviving elements left over the spaces left by removed elements
		final boolean anyToRemove = removeCount > 0;
		if (anyToRemove) {
			final int newSize = size - removeCount;
			for (int i = 0, j = 0; (i < size) && (j < newSize); i++, j++) {
				i = removeSet.nextClearBit(i);
				vals[j] = vals[i];
			}
			for (int k = newSize; k < size; k++) {
				vals[k] = 0; // Let gc do its work
			}
			this.size = newSize;
			if (modCount != expectedModCount) {
				throw new ConcurrentModificationException();
			}
			modCount++;
		}

		return anyToRemove;
	}

	/**
	 * Removes from this list all of the elements whose index is between {@code fromIndex}, inclusive, and {@code toIndex}, exclusive.
	 * Shifts any succeeding elements to the left (reduces their index). This call shortens the list by {@code (toIndex - fromIndex)}
	 * elements. (If {@code toIndex==fromIndex}, this operation has no effect.)
	 *
	 * @throws IndexOutOfBoundsException
	 *             if {@code fromIndex} or {@code toIndex} is out of range ({@code fromIndex < 0 ||
	 *          fromIndex >= size() ||
	 *          toIndex > size() ||
	 *          toIndex < fromIndex})
	 */
	protected void removeRange(int fromIndex, int toIndex) {
		modCount++;
		int numMoved = size - toIndex;
		System.arraycopy(vals, toIndex, vals, fromIndex, numMoved);

		// clear to let GC do its work
		int newSize = size - (toIndex - fromIndex);
		for (int i = newSize; i < size; i++) {
			vals[i] = 0;
		}
		size = newSize;
	}

	@SuppressWarnings("unchecked")
	public void replaceAll(UnaryOperator<Short> operator) {
		Objects.requireNonNull(operator);
		final int expectedModCount = modCount;
		final int size = this.size;
		for (int i = 0; modCount == expectedModCount && i < size; i++) {
			vals[i] = operator.apply(vals[i]);
		}
		if (modCount != expectedModCount) {
			throw new ConcurrentModificationException();
		}
		modCount++;
	}

	/**
	 * Retains only the elements in this list that are contained in the specified collection. In other words, removes from this list all of
	 * its elements that are not contained in the specified collection.
	 *
	 * @param c
	 *            collection containing elements to be retained in this list
	 * @return {@code true} if this list changed as a result of the call
	 * @throws ClassCastException
	 *             if the class of an element of this list is incompatible with the specified collection
	 *             (<a href="Collection.html#optional-restrictions">optional</a>)
	 * @throws NullPointerException
	 *             if this list contains a null element and the specified collection does not permit null elements
	 *             (<a href="Collection.html#optional-restrictions">optional</a>), or if the specified collection is null
	 * @see Collection#contains(Object)
	 */
	public boolean retainAll(Collection<Short> c) {
		Objects.requireNonNull(c);
		return batchRemove(c, true);
	}

	public void reverse() {
		final int expectedModCount = modCount;

		int mid = size / 2;
		for (int i = 0; i < mid; i++) {
			swap(i, size - 1 - i);
		}

		if (modCount != expectedModCount) {
			throw new ConcurrentModificationException();
		}
		modCount++;
	}

	/**
	 * Replaces the element at the specified position in this list with the specified element.
	 *
	 * @param index
	 *            index of the element to replace
	 * @param element
	 *            element to be stored at the specified position
	 * @return the element previously at the specified position
	 * @throws IndexOutOfBoundsException
	 *             {@inheritDoc}
	 */
	public short set(int index, short element) {
		rangeCheck(index);

		short oldValue = value(index);
		vals[index] = element;
		return oldValue;
	}

	/**
	 * Returns the number of elements in this list.
	 *
	 * @return the number of elements in this list
	 */
	public int size() {
		return size;
	}

	public void sort(boolean descending) {
		final int expectedModCount = modCount;

		Arrays.sort(vals, 0, size);

		if (modCount != expectedModCount) {
			throw new ConcurrentModificationException();
		}
		modCount++;

		if (descending) {
			reverse();
		}
	}

	/**
	 * Creates a <em><a href="Spliterator.html#binding">late-binding</a></em> and <em>fail-fast</em> {@link Spliterator} over the elements
	 * in this list.
	 *
	 * <p>
	 * The {@code Spliterator} reports {@link Spliterator#SIZED}, {@link Spliterator#SUBSIZED}, and {@link Spliterator#ORDERED}. Overriding
	 * implementations should document the reporting of additional characteristic values.
	 *
	 * @return a {@code Spliterator} over the elements in this list
	 * @since 1.8
	 */
	public Spliterator<Short> spliterator() {
		return new ArrayListSpliterator<>(this, 0, -1, 0);
	}

	/**
	 * Returns a view of the portion of this list between the specified {@code fromIndex}, inclusive, and {@code toIndex}, exclusive. (If
	 * {@code fromIndex} and {@code toIndex} are equal, the returned list is empty.) The returned list is backed by this list, so
	 * non-structural changes in the returned list are reflected in this list, and vice-versa. The returned list supports all of the
	 * optional list operations.
	 *
	 * <p>
	 * This method eliminates the need for explicit range operations (of the sort that commonly exist for arrays). Any operation that
	 * expects a list can be used as a range operation by passing a subList view instead of a whole list. For example, the following idiom
	 * removes a range of elements from a list:
	 * 
	 * <pre>
	 * list.subList(from, to).clear();
	 * </pre>
	 * 
	 * Similar idioms may be constructed for {@link #indexOf(Object)} and {@link #lastIndexOf(Object)}, and all of the algorithms in the
	 * {@link Collections} class can be applied to a subList.
	 *
	 * <p>
	 * The semantics of the list returned by this method become undefined if the backing list (i.e., this list) is <i>structurally
	 * modified</i> in any way other than via the returned list. (Structural modifications are those that change the size of this list, or
	 * otherwise perturb it in such a fashion that iterations in progress may yield incorrect results.)
	 *
	 * @throws IndexOutOfBoundsException
	 *             {@inheritDoc}
	 * @throws IllegalArgumentException
	 *             {@inheritDoc}
	 */
	public ShortArray subArray(int fromIndex, int toIndex) {
		subListRangeCheck(fromIndex, toIndex, size);
		int size = toIndex - fromIndex;
		ShortArray ret = new ShortArray(size);
		for (int i = fromIndex; i < toIndex; i++) {
			ret.add(get(i));
		}
		return ret;
	}

	public void swap(int i, int j) {
		short v1 = get(i);
		short v2 = get(j);

		set(i, v2);
		set(j, v1);
	}

	/**
	 * Returns an array containing all of the elements in this list in proper sequence (from first to last element).
	 *
	 * <p>
	 * The returned array will be "safe" in that no references to it are maintained by this list. (In other words, this method must allocate
	 * a new array). The caller is thus free to modify the returned array.
	 *
	 * <p>
	 * This method acts as bridge between array-based and collection-based APIs.
	 *
	 * @return an array containing all of the elements in this list in proper sequence
	 */
	public Object[] toArray() {
		// return Arrays.copyOf(vals, size);

		return null;
	}

	/**
	 * Returns an array containing all of the elements in this list in proper sequence (from first to last element); the runtime type of the
	 * returned array is that of the specified array. If the list fits in the specified array, it is returned therein. Otherwise, a new
	 * array is allocated with the runtime type of the specified array and the size of this list.
	 *
	 * <p>
	 * If the list fits in the specified array with room to spare (i.e., the array has more elements than the list), the element in the
	 * array immediately following the end of the collection is set to <tt>null</tt>. (This is useful in determining the length of the list
	 * <i>only</i> if the caller knows that the list does not contain any null elements.)
	 *
	 * @param a
	 *            the array into which the elements of the list are to be stored, if it is big enough; otherwise, a new array of the same
	 *            runtime type is allocated for this purpose.
	 * @return an array containing the elements of the list
	 * @throws ArrayStoreException
	 *             if the runtime type of the specified array is not a supertype of the runtime type of every element in this list
	 * @throws NullPointerException
	 *             if the specified array is null
	 */
	@SuppressWarnings("unchecked")
	public <T> T[] toArray(T[] a) {
		// if (a.length < size)
		// // Make a new array of a's runtime type, but my contents:
		// return (T[]) Arrays.copyOf(vals, size, a.getClass());
		// System.arraycopy(vals, 0, a, 0, size);
		// if (a.length > size)
		// a[size] = null;
		// return a;

		return null;
	}

	public String toString() {
		return toString(30, false);
	}

	public String toString(int size, boolean sparse) {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("(%d/%d) ", size(), vals.length));

		for (int i = 0; i < size(); i++) {
			if (i == size) {
				sb.append("...");
				break;
			}

			int v = value(i);
			if (sparse) {
				if (v != 0) {
					sb.append(String.format("%d:%d ", i, v));
				}
			} else {
				sb.append(v);

				if (i != size() - 1) {
					sb.append(", ");
				}
			}
		}
		return "[" + sb.toString().trim() + "]";
	}

	/**
	 * Trims the capacity of this <tt>ArrayList</tt> instance to be the list's current size. An application can use this operation to
	 * minimize the storage of an <tt>ArrayList</tt> instance.
	 */
	public void trimToSize() {
		modCount++;
		if (size < vals.length) {
			vals = (size == 0) ? EMPTY_VALUES : Arrays.copyOf(vals, size);
		}
	}

	public short value(int index) {
		return vals[index];
	}

	public short[] values() {
		return vals;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(size());
		for (int i = 0; i < size(); i++) {
			oos.writeShort(get(i));
		}
	}

	public void writeObject(String fileName) throws Exception {
		System.out.printf("write at [%s]\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
