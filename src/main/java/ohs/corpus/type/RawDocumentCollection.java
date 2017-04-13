package ohs.corpus.type;

import java.io.File;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.WeakHashMap;

import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayMath;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.ListList;
import ohs.types.number.IntegerArray;
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

	public RawDocumentCollection(FileChannel fc, ListList<String> attrData, ListList<Boolean> flagData, LongArray starts, IntegerArray lens,
			ShortArray types, WeakHashMap<Integer, ArrayList<String>> cache) {
		super();
		this.fc = fc;
		this.attrData = attrData;
		this.flagData = flagData;
		this.starts = starts;
		this.lens = lens;
		this.types = types;
		this.cache = cache;
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
				types, cache);
	}

	public ArrayList<String> get(int i) throws Exception {
		ArrayList<String> ret = cache.get(i);
		if (ret == null) {
			long start = starts.get(i);
			fc.position(start);

			ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);

			List<Boolean> flags = flagData.get(types.get(i), false);

			ret = Generics.newArrayList(data.size());

			for (int j = 0; j < data.size(); j++) {
				String val = flags.get(j) ? DataCompression.decodeToString(data.get(j)) : new String(data.get(j).values());
				ret.add(val);
			}
			synchronized (cache) {
				cache.put(i, ret);
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

	public ListList<String> getRange(int[] range) throws Exception {
		return getRange(range[0], range[1]);
	}

	public ListList<String> getRange(int i, int j) throws Exception {
		long start = starts.get(i);
		int size1 = ArrayMath.sum(lens.values(), i, j);

		fc.position(start);
		ByteArray data = FileUtils.readByteArray(fc, size1);
		ByteBufferWrapper buf = new ByteBufferWrapper(data);

		ListList<String> ret = Generics.newListList(j - i);

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
		}
		return ret;
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

	public ShortArray getTypes() {
		return types;
	}

	public ListList<String> getValues(int[] range) throws Exception {
		return getRange(range[0], range[1]);
	}

	public int size() {
		return starts.size();
	}

}
