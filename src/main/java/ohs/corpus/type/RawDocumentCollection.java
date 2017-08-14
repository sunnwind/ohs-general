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

import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayMath;
import ohs.types.generic.Counter;
import ohs.types.generic.ListList;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.types.number.LongArray;
import ohs.types.number.ShortArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class RawDocumentCollection {

	public static final String DATA_NAME = "raw_docs.ser";

	public static final String META_NAME = "raw_docs_meta.ser";

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// {
		// RawDocumentCollection rdc = new
		// RawDocumentCollection(MIRPath.WIKI_COL_DC_DIR);
		// System.out.println(rdc.size());
		// // System.out.println(rdc.getText(0));
		// // System.out.println(rdc.getText(10));
		//
		// for (int i = 0; i < rdc.size(); i++) {
		// Map<String, String> m = rdc.getMap(i);
		// String title = m.get("title");
		// String content = m.get("content");
		//
		// List<String> sents = StrUtils.split("\n", content);
		//
		// if (sents.size() > 1) {
		//
		// String sent = sents.get(1);
		//
		// if (sent.contains("also known as")) {
		// System.out.println(title);
		// System.out.println(sent);
		// }
		// }
		// }
		//
		// rdc.close();
		// }

		// {
		// RawDocumentCollection rdc = new
		// RawDocumentCollection(MIRPath.OHSUMED_COL_DC_DIR);
		// System.out.println(rdc.toString());
		//
		// for (int i = 0; i < 10; i++) {
		// String s = rdc.getText(i);
		//
		// System.out.println(s);
		// System.out.println();
		// }
		// }
		//
		{
			RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.TREC_PM_2017_COL_MEDLINE_DC_DIR);
			System.out.println(rdc.toString());
		}

		{
			RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.TREC_PM_2017_COL_CLINICAL_DC_DIR);
			System.out.println(rdc.toString());
		}

		{
			RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);
			System.out.println(rdc.toString());
		}

		{
			RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.WIKI_COL_DC_DIR);
			System.out.println(rdc.toString());
		}

		{
			RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.DATA_DIR + "scopus/col/dc/");
			System.out.println(rdc.toString());
		}

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

		// {
		// RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.DATA_DIR +
		// "merged/col/dc/");
		// ListList<String> atrData = rdc.getAttrData();
		// IntegerArrayMatrix ranges = rdc.getTypeRanges();
		// System.out.println(atrData);
		// System.out.println(ranges);
		// System.out.println();
		// }

		System.out.println("process ends.");
	}

	private WeakHashMap<Integer, ArrayList<String>> cache = Generics.newWeakHashMap();

	private FileChannel fc;

	private ListList<String> attrData;

	private ListList<Boolean> flagData;

	private LongArray starts;

	private IntegerArray sizes;

	private IntegerArray lens;

	private File dataDir;

	private boolean use_cache = true;

	private RawDocumentCollection(FileChannel fc, ListList<String> attrData, ListList<Boolean> flagData,
			LongArray starts, IntegerArray lens, IntegerArray sizes, WeakHashMap<Integer, ArrayList<String>> cache,
			boolean use_cache) {
		super();
		this.fc = fc;
		this.attrData = attrData;
		this.flagData = flagData;
		this.starts = starts;
		this.lens = lens;
		this.sizes = sizes;
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
				sizes = DataCompression.decodeToIntegerArray(data.get(i++));

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
		return new RawDocumentCollection(FileUtils.openFileChannel(new File(dataDir, DATA_NAME), "r"), attrData,
				flagData, starts, lens, sizes, cache, use_cache);
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
				// data = new ByteBufferWrapper(FileUtils.readByteArray(fc,
				// len)).readByteArrayMatrix();
				data = FileUtils.readByteArrayMatrix(fc, len);
			}

			int cseq = getColSeq(dseq);
			List<Boolean> flags = flagData.get(cseq, false);

			ret = Generics.newArrayList(data.size());

			for (int j = 0; j < data.size(); j++) {
				String val = flags.get(j) ? DataCompression.decodeToString(data.get(j))
						: new String(data.get(j).values());
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

	public int getColSeq(int dseq) {
		int ret = 0;
		if (sizes.size() > 1) {
			int tmp = 0;
			for (int i = 0; i < sizes.size(); i++) {
				int s = tmp;
				int e = s + sizes.get(i);
				if (dseq >= s && dseq < e) {
					ret = i;
					break;
				}
				tmp = e;
			}
		}
		return ret;
	}

	public FileChannel getFileChannel() {
		return fc;
	}

	public HashMap<String, String> getMap(int i) throws Exception {
		HashMap<String, String> ret = Generics.newHashMap();
		List<String> attrs = attrData.get(getColSeq(i), false);
		List<String> vals = get(i);
		for (int j = 0; j < attrs.size(); j++) {
			ret.put(attrs.get(j), vals.get(j));
		}
		return ret;
	}

	public Map<String, String> getMap(List<String> vals) {
		HashMap<String, String> ret = Generics.newHashMap();
		List<String> attrs = attrData.get(0, false);
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
				int size2 = buf.readInteger();
				ByteArrayMatrix doc = buf.readByteArrayMatrix();
				int type = getColSeq(k);
				List<Boolean> flags = flagData.get(type, false);
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

	public IntegerArray getSizes() {
		return sizes;
	}

	public String getText(int i) throws Exception {
		List<String> attrs = attrData.get(getColSeq(i), false);
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
		IntegerArrayMatrix ret = new IntegerArrayMatrix(sizes.size());
		int s = 0;
		for (int i = 0; i < sizes.size(); i++) {
			int e = s + sizes.get(i);
			ret.add(new IntegerArray(new int[] { s, e }));
			s = e;
		}
		return ret;
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

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer("==================");
		sb.append(String.format("\nsize:\t%d", size()));
		sb.append("\n" + attrData.toString());
		for (int i = 0; i < sizes.size(); i++) {
			sb.append(String.format("\n%d:\t%d", i, sizes.get(i)));
		}
		return sb.toString();
	}

}
