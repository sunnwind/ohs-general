package ohs.corpus.type;

import java.io.File;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayMath;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.number.IntegerArray;
import ohs.types.number.LongArray;
import ohs.utils.Generics;
import ohs.utils.Timer;

public class DocumentIdMap {

	public static final String DATA_NAME = "doc_id_map.ser";

	public static final String META_NAME = "doc_id_map_meta.ser";

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		{
			String[] dirs = { MIRPath.OHSUMED_COL_DC_DIR, MIRPath.TREC_CDS_2014_COL_DC_DIR, MIRPath.TREC_CDS_2016_COL_DC_DIR,
					MIRPath.WIKI_COL_DC_DIR, MIRPath.BIOASQ_COL_DC_DIR };

			for (int j = 1; j < dirs.length; j++) {
				String dir = dirs[j];

				System.out.println(dir);

				DocumentIdMap dc = new DocumentIdMap(dir);
				int[][] ranges = BatchUtils.getBatchRanges(dc.size(), 500);

				Timer timer = Timer.newTimer();

				for (int i = 0; i < ranges.length; i++) {
					Timer timer2 = Timer.newTimer();
					dc.getRange(ranges[i][0], ranges[i][1]);
					// System.out.println(timer2.stop());
				}

				System.out.println(timer.stop());

				dc = new DocumentIdMap(dir);
				timer = Timer.newTimer();

				Timer timer2 = Timer.newTimer();

				for (int i = 0; i < dc.size(); i++) {
					if ((i + 1) % 10000 == 0 || i == dc.size() - 1) {
						// System.out.println(timer2.stop());
						timer2 = Timer.newTimer();
					}
					dc.get(i);
				}

				System.out.println(timer.stop());
				System.out.println();
			}
		}

		System.out.println("process ends.");
	}

	private Map<Integer, String> cache = Generics.newWeakHashMap();;

	private FileChannel fc;

	private File dataDir;

	private LongArray starts;

	private IntegerArray lens;

	private boolean encode = false;

	public DocumentIdMap() {

	}

	public DocumentIdMap(FileChannel fc, LongArray starts, IntegerArray lens, Map<Integer, String> cache) throws Exception {
		this.fc = fc;
		this.starts = starts;
		this.lens = lens;
		this.cache = cache;
	}

	public DocumentIdMap(String dataDir) throws Exception {
		this.dataDir = new File(dataDir);

		{
			FileChannel fc = FileUtils.openFileChannel(new File(dataDir, META_NAME), "r");
			ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);
			fc.close();

			int i = 0;
			starts = DataCompression.decodeToLongArray(data.get(i++));
			lens = DataCompression.decodeToIntegerArray(data.get(i++));
			encode = data.get(i++).get(0) == 1 ? true : false;

			DataCompression.decodeGaps(starts.values());
		}

		fc = FileUtils.openFileChannel(new File(dataDir, DATA_NAME), "r");
	}

	public void close() throws Exception {
		fc.close();
	}

	public DocumentIdMap copyShallow() throws Exception {
		return new DocumentIdMap(FileUtils.openFileChannel(new File(dataDir, DATA_NAME), "r"), starts, lens, cache);
	}

	public String get(int i) throws Exception {
		String ret = null;

		synchronized (cache) {
			ret = cache.get(i);
		}

		if (ret == null) {
			ByteArray data = null;

			synchronized (fc) {
				long start = starts.get(i);
				fc.position(start);
				data = FileUtils.readByteArray(fc);
			}

			if (encode) {
				ret = DataCompression.decodeToString(data);
			} else {
				ret = new String(data.values());
			}

			synchronized (cache) {
				cache.put(i, ret);
			}
		}

		return ret;
	}

	public List<String> get(int[] is) throws Exception {
		List<String> ret = Generics.newArrayList(is.length);
		for (int i : is) {
			ret.add(get(i));
		}
		return ret;
	}

	public File getDataDir() {
		return dataDir;
	}

	public int getDocLength(int i) {
		return lens.get(i);
	}

	public FileChannel getFileChannel() {
		return fc;
	}

	public List<String> getRange(int i, int j) throws Exception {
		int size = j - i;

		Map<Integer, String> found = Generics.newHashMap(size);
		Set<Integer> notFound = Generics.newHashSet(size);

		for (int k = i; k < j; k++) {
			String p = null;

			synchronized (cache) {
				p = cache.get(k);
			}

			if (p == null) {
				notFound.add(k);
			} else {
				found.put(k, p);
			}
		}

		List<String> ret = Generics.newArrayList(size);

		if (found.size() == 0) {
			ByteArray data = null;

			synchronized (fc) {
				fc.position(starts.get(i));
				data = FileUtils.readByteArray(fc, ArrayMath.sum(lens.values(), i, j));
			}

			ByteBufferWrapper buf = new ByteBufferWrapper(data);

			for (int k = i; k < j; k++) {
				ByteArrayMatrix sub = buf.readByteArrayMatrix();

				String docid = null;
				int u = 0;
				if (encode) {
					docid = DataCompression.decodeToString(sub.get(u++));
				} else {
					docid = new String(sub.get(u++).values());
				}
				String p = docid;
				ret.add(p);

				synchronized (cache) {
					cache.put(k, p);
				}
			}
		} else {
			if (found.size() != size) {
				for (int k : notFound) {
					found.put(k, get(k));
				}
			}
			for (int k = i; k < j; k++) {
				ret.add(found.get(k));
			}
		}
		return ret;
	}

	public List<String> getRange(int[] range) throws Exception {
		return getRange(range[0], range[1]);
	}

	public LongArray getStarts() {
		return starts;
	}

	public int size() {
		return starts.size();
	}

}
