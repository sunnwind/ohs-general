package ohs.corpus.type;

import java.io.File;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;

import org.apache.xalan.xsltc.compiler.sym;

import kr.co.shineware.nlp.komoran.a.a;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayMath;
import ohs.types.generic.Counter;
import ohs.types.generic.ListList;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.types.number.LongArray;
import ohs.types.number.ShortArray;
import ohs.utils.Generics;

public class RawDocumentCollection {

	public static final String DATA_NAME = "raw_docs.ser";

	public static final String META_NAME = "raw_docs_meta.ser";

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// {
		// RawDocumentCollection rdc = new
		// RawDocumentCollection(MIRPath.WIKI_COL_DC_DIR);
		// System.out.println(rdc.size());
		// System.out.println(rdc.getAttrValueText(0));
		// System.out.println(rdc.getAttrValueText(20913));
		// }
		//
		// {
		// RawDocumentCollection rdc = new
		// RawDocumentCollection(KPPath.COL_DC_DIR);
		// System.out.println(rdc.size());
		// System.out.println(rdc.getAttrValueText(0));
		// System.out.println(rdc.getAttrValueText(20913));
		//
		// Counter<String> c = Generics.newCounter();
		//
		// for (int i = 0; i < rdc.size(); i++) {
		// List<String> vals = rdc.getValues(i);
		// c.incrementCount(vals.get(0), 1);
		// }
		//
		// System.out.println(c.toStringSortedByValues(true, true, c.size(),
		// "\t"));
		//
		// }

		// {
		// RawDocumentCollection rdc = new
		// RawDocumentCollection(MIRPath.BIOASQ_COL_DC_DIR);
		// System.out.println(rdc.size());
		// System.out.println(rdc.getAttrValueText(0));
		// System.out.println(rdc.getAttrValueText(20913));
		// }
		//
		// {
		// RawDocumentCollection rdc = new
		// RawDocumentCollection(KPPath.COL_DC_DIR);
		// System.out.println(rdc.size());
		// System.out.println(rdc.getAttrValueText(0));
		// System.out.println(rdc.getAttrValueText(20913));
		// }

		{
			RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.DATA_DIR + "merged/col/dc/");
			ListList<String> atrData = rdc.getAttrData();
			IntegerArrayMatrix ranges = rdc.getTypeRanges();
			System.out.println(atrData);
			System.out.println(ranges);
			System.out.println();

		}

		System.out.println("process ends.");
	}

	private WeakHashMap<Integer, ArrayList<String>> cache = Generics.newWeakHashMap();

	private FileChannel fc;

	private ListList<String> attrData;

	private ListList<Boolean> flagData;

	private LongArray starts;

	private ShortArray types;

	private IntegerArray lens;

	private File dataDir;

	private boolean use_cache = true;

	private RawDocumentCollection(FileChannel fc, ListList<String> attrData, ListList<Boolean> flagData, LongArray starts,
			IntegerArray lens, ShortArray types, WeakHashMap<Integer, ArrayList<String>> cache, boolean use_cache) {
		super();
		this.fc = fc;
		this.attrData = attrData;
		this.flagData = flagData;
		this.starts = starts;
		this.lens = lens;
		this.types = types;
		this.cache = cache;
		this.use_cache = use_cache;
	}

	public RawDocumentCollection(String dataDir) throws Exception {
		this.dataDir = new File(dataDir);

		fc = FileUtils.openFileChannel(new File(dataDir, DATA_NAME), "r");

		{
			FileChannel fc2 = FileUtils.openFileChannel(new File(dataDir, META_NAME), "r");

			{
				int i = 0;
				ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc2);
				starts = DataCompression.decodeToLongArray(data.get(i++));
				lens = DataCompression.decodeToIntegerArray(data.get(i++));
				types = DataCompression.decodeToShortArray(data.get(i++));

				DataCompression.decodeGaps(starts);
			}

			{
				ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc2);
				attrData = Generics.newListList(data.size());

				for (int i = 0; i < data.size(); i++) {
					List<String> attrs = Generics.newArrayList();
					for (String attr : DataCompression.decodeToString(data.get(i)).split("\t")) {
						attrs.add(attr);
					}
					attrData.add(attrs);
				}
			}

			{
				ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc2);
				flagData = Generics.newListList(data.size());

				for (int i = 0; i < data.size(); i++) {
					List<Boolean> flags = Generics.newArrayList(data.get(i).size());
					ByteBufferWrapper buf = new ByteBufferWrapper(data.get(i));
					for (boolean flag : buf.readBooleans()) {
						flags.add(flag);
					}
					flagData.add(flags);
				}
			}
			fc2.close();
		}
	}

	public void close() throws Exception {
		fc.close();
	}

	public RawDocumentCollection copyShallow() throws Exception {
		return new RawDocumentCollection(FileUtils.openFileChannel(new File(dataDir, DATA_NAME), "r"), attrData, flagData, starts, lens,
				types, cache, use_cache);
	}

	public ArrayList<String> get(int dseq) throws Exception {
		ArrayList<String> ret = null;

		if (use_cache) {
			synchronized (cache) {
				ret = cache.get(dseq);
			}
		}

		if (ret == null) {
			ByteArrayMatrix data = null;

			synchronized (fc) {
				long start = starts.get(dseq);
				fc.position(start);
				int len = lens.get(dseq);
				data = new ByteBufferWrapper(FileUtils.readByteArray(fc, len)).readByteArrayMatrix();
			}

			List<Boolean> flags = flagData.get(types.get(dseq), false);

			ret = Generics.newArrayList(data.size());

			for (int j = 0; j < data.size(); j++) {
				String val = flags.get(j) ? DataCompression.decodeToString(data.get(j)) : new String(data.get(j).values());
				ret.add(val);
			}

			if (use_cache) {
				synchronized (cache) {
					cache.put(dseq, ret);
				}
			}
		}
		return ret;
	}

	public ListList<String> getAttrData() {
		return attrData;
	}

	public FileChannel getFileChannel() {
		return fc;
	}

	public HashMap<String, String> getMap(int i) throws Exception {
		HashMap<String, String> ret = Generics.newHashMap();
		List<String> attrs = attrData.get(types.get(i), false);
		List<String> vals = get(i);
		for (int j = 0; j < attrs.size(); j++) {
			ret.put(attrs.get(j), vals.get(j));
		}
		return ret;
	}

	public Map<String, String> getMap(List<String> vals) {
		HashMap<String, String> ret = Generics.newHashMap();
		List<String> attrs = attrData.get(types.get(0), false);
		for (int j = 0; j < attrs.size(); j++) {
			ret.put(attrs.get(j), vals.get(j));
		}
		return ret;
	}

	public ListList<String> getRange(int i, int j) throws Exception {
		int size = ArrayMath.sum(lens.values(), i, j);

		Map<Integer, List<String>> found = Generics.newHashMap(size);
		Set<Integer> notFound = Generics.newHashSet(size);

		for (int k = i; k < j; k++) {
			ArrayList<String> p = null;

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

		ListList<String> ret = Generics.newListList(size);

		if (found.size() == 0) {
			ByteArray data = null;

			synchronized (fc) {
				long start = starts.get(i);
				fc.position(start);
				data = FileUtils.readByteArray(fc, size);
			}

			ByteBufferWrapper buf = new ByteBufferWrapper(data);

			for (int k = i; k < j; k++) {
				ByteArrayMatrix doc = buf.readByteArrayMatrix();
				List<Boolean> flags = flagData.get(types.get(k), false);
				ArrayList<String> vals = Generics.newArrayList(doc.size());

				for (int l = 0; l < doc.size(); l++) {
					ByteArray sub = doc.get(l);
					String val = flags.get(l) ? DataCompression.decodeToString(sub) : new String(sub.values());
					vals.add(val);
				}
				ret.add(vals);

				if (use_cache) {
					synchronized (cache) {
						cache.put(k, vals);
					}
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

	public ListList<String> getRange(int[] range) throws Exception {
		return getRange(range[0], range[1]);
	}

	public String getText(int i) throws Exception {
		List<String> attrs = attrData.get(types.get(i), false);
		List<String> vals = get(i);
		StringBuffer sb = new StringBuffer();

		for (int j = 0; j < attrs.size(); j++) {
			sb.append(String.format("%d,\t%s,\t%s", j + 1, attrs.get(j), vals.get(j)));
			if (j != attrs.size() - 1) {
				sb.append("\n");
			}
		}
		return sb.toString();
	}

	public IntegerArrayMatrix getTypeRanges() {
		Counter<Integer> typeCnts = Generics.newCounter();
		for (int i = 0; i < types.size(); i++) {
			typeCnts.incrementCount((int) types.get(i), 1);
		}

		List<Integer> keys = Generics.newArrayList(typeCnts.keySet());
		Collections.sort(keys);

		IntegerArrayMatrix ret = new IntegerArrayMatrix(keys.size());

		int s = 0;
		for (int i = 0; i < keys.size(); i++) {
			int type = keys.get(i);
			int size = (int) typeCnts.getCount(type);
			int e = s + size + 1;
			ret.add(new IntegerArray(new int[] { s, e }));
			s = e;
		}
		return ret;
	}

	public ShortArray getTypes() {
		return types;
	}

	public ListList<String> getValues(int[] range) throws Exception {
		return getRange(range[0], range[1]);
	}

	public void setUseCache(boolean use_cache) {
		this.use_cache = use_cache;
	}

	public int size() {
		return starts.size();
	}

}
