package ohs.ir.search.index;

import java.io.File;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.corpus.type.DataCompression;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.types.number.LongArray;
import ohs.utils.Generics;
import ohs.utils.Timer;

public class DiskInvertedIndex extends InvertedIndex {

	public static final String DATA_NAME = "posting.ser";

	public static final String META_NAME = "posting_meta.ser";

	private Map<Integer, PostingList> cache = Generics.newWeakHashMap();

	private File dataDir;

	private FileChannel fc;

	private LongArray starts;

	private IntegerArray lens;

	private boolean encode = false;

	private boolean use_cache = true;

	private DiskInvertedIndex(FileChannel fc, LongArray starts, IntegerArray lens, int doc_cnt,
			Map<Integer, PostingList> cache, Vocab vocab, boolean use_cache, boolean print_log) {
		this.fc = fc;
		this.starts = starts;
		this.lens = lens;
		this.doc_cnt = doc_cnt;
		this.cache = cache;
		this.vocab = vocab;
		this.use_cache = use_cache;
	}

	public DiskInvertedIndex(FileChannel fc, LongArray starts, IntegerArray lens, int doc_cnt, Vocab vocab) {
		this(fc, starts, lens, doc_cnt, Generics.newWeakHashMap(), vocab, true, false);
	}

	public DiskInvertedIndex(String dataDir) throws Exception {
		this(dataDir, null);
	}

	public DiskInvertedIndex(String dataDir, Vocab vocab) throws Exception {
		this.dataDir = new File(dataDir);

		{
			FileChannel fc = FileUtils.openFileChannel(new File(dataDir, META_NAME), "r");
			ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);
			fc.close();

			int i = 0;
			doc_cnt = ByteArrayUtils.toInteger(data.get(i++));
			starts = DataCompression.decodeToLongArray(data.get(i++));
			lens = DataCompression.decodeToIntegerArray(data.get(i++));
			DataCompression.decodeGaps(starts);
		}

		fc = FileUtils.openFileChannel(new File(dataDir, DATA_NAME), "r");

		this.vocab = vocab;
	}

	public void close() throws Exception {
		fc.close();
	}

	public DiskInvertedIndex copyShallow() throws Exception {
		return new DiskInvertedIndex(FileUtils.openFileChannel(new File(dataDir, DATA_NAME), "r"), starts, lens,
				doc_cnt, cache, vocab, use_cache, print_log);
	}

	public FileChannel getFileChannel() {
		return fc;
	}

	public PostingList getPostingList(int w) throws Exception {
		Timer timer = Timer.newTimer();

		PostingList ret = new PostingList(-1, new IntegerArray(), new IntegerMatrix());

		boolean is_cached = false;

		if (w < 0 || w >= starts.size()) {
			return null;
		}

		if (use_cache) {
			synchronized (cache) {
				ret = cache.get(w);
			}
		}

		if (ret == null) {
			ByteArray data = null;

			synchronized (fc) {
				long start = starts.get(w);
				if (start < 0) {
					return null;
				}
				fc.position(start);

				int len = lens.get(w);

				ret = PostingList.readPostingList(fc);

				IntegerArray dseqs = ret.getDocSeqs();

				for (int i = 0; i < dseqs.size(); i++) {
					int dseq = dseqs.get(i);
					if (dseq < 0 || dseq >= doc_cnt) {
						System.out.println();
					}
				}

				// data = FileUtils.readByteArray(fc, len);
			}

			// ret = PostingList.readPostingList(fc);
			// ret = PostingList.toPostingList(new
			// ByteBufferWrapper(data).readByteArrayMatrix(), encode);

			if (use_cache) {
				synchronized (cache) {
					cache.put(w, ret);
				}
			}
		} else {
			is_cached = true;
		}

		if (print_log) {
			String word = vocab == null ? "null" : vocab.getObject(w);
			System.out.printf("PL: cached=[%s], word=[%s], %s, time=[%s]\n", is_cached, word, is_cached, ret,
					timer.stop());
		}
		return ret;
	}

	public List<PostingList> getPostingLists(int i, int j) throws Exception {
		int size = j - i;

		List<PostingList> ret = Generics.newArrayList(size);

		Map<Integer, PostingList> found = Generics.newHashMap(size);
		Set<Integer> notFound = Generics.newHashSet(size);

		for (int k = i; k < j; k++) {
			PostingList p = null;

			if (use_cache) {
				synchronized (cache) {
					p = cache.get(k);
				}
			}

			if (p == null) {
				notFound.add(k);
			} else {
				found.put(k, p);
			}
		}

		if (found.size() == 0) {
			ByteArray data = null;

			synchronized (fc) {
				fc.position(starts.get(i));
				data = FileUtils.readByteArray(fc, ArrayMath.sum(lens.values(), i, j));
			}

			ByteBufferWrapper buf = new ByteBufferWrapper(data);

			for (int k = i; k < j; k++) {
				ByteArrayMatrix sub = buf.readByteArrayMatrix();
				PostingList pl = PostingList.toPostingList(sub);
				ret.add(pl);

				if (use_cache) {
					synchronized (cache) {
						cache.put(k, pl);
					}
				}
			}
		} else {
			if (found.size() != size) {
				for (int k : notFound) {
					found.put(k, getPostingList(k));
				}
			}
			for (int k = i; k < j; k++) {
				ret.add(found.get(k));
			}
		}
		return ret;
	}

	public void setUseCache(boolean use_cache) {
		this.use_cache = use_cache;
	}

	public int size() {
		return starts.size();
	}

}
