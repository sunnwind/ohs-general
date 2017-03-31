package ohs.eden.keyphrase.mine;

import java.nio.channels.FileChannel;
import java.util.List;

import ohs.corpus.search.index.Posting;
import ohs.corpus.type.DataCompression;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;

public class SPostingList {

	private static IntegerArrayMatrix partition(IntegerArray a, int chunk_size) {
		int window_size = chunk_size / Integer.BYTES;
		List<IntegerArray> ret = Generics.newLinkedList();
		int s = 0;
		while (s < a.size()) {
			int e = Math.min(a.size(), s + window_size);
			ret.add(a.subArray(s, e));
			s = e;
		}
		return new IntegerArrayMatrix(ret);
	}

	public static SPostingList readPostingList(FileChannel fc, boolean encode) throws Exception {
		SPostingList ret = null;
		if (fc.position() < fc.size()) {
			ret = toPostingList(FileUtils.readByteArrayMatrix(fc), encode);
		}
		return ret;
	}

	public static List<SPostingList> readPostingLists(FileChannel fc, boolean encode) throws Exception {
		List<SPostingList> ret = Generics.newLinkedList();
		if (fc.position() < fc.size()) {
			ret.add(toPostingList(FileUtils.readByteArrayMatrix(fc), encode));
		}
		return ret;
	}

	public static int sizeOfByteBuffer(SPostingList pl) {
		int ret = Integer.BYTES;
		ret += Integer.BYTES * pl.getDocseqs().size() * 2;
		ret += Integer.BYTES * pl.getPosData().sizeOfEntries();
		ret += Integer.BYTES * pl.getPosData().size();
		return ret;
	}

	public static ByteArrayMatrix toByteArrayMatrix(SPostingList pl, boolean encode) throws Exception {
		String phrs = pl.getPhrase();
		IntegerArray dseqs = pl.getDocseqs();
		IntegerArrayMatrix posData = pl.getPosData();

		int min_encode_len = 20;

		int max_buf_size = FileUtils.DEFAULT_BUF_SIZE;

		List<ByteArray> data = Generics.newLinkedList();

		if (encode) {
			byte encode_dseqs = 0;

			if (dseqs.size() > min_encode_len) {
				encode_dseqs = 1;
			}

			data.add(new ByteArray(phrs.getBytes()));
			data.add(new ByteArray(new byte[] { encode_dseqs }));
			data.add(ByteArrayUtils.toByteArray(dseqs.size()));

			if (encode_dseqs == 1) {
				DataCompression.encodeGaps(dseqs);
				data.add(DataCompression.encode(dseqs));
			} else {
				data.add(ByteArrayUtils.toByteArray(dseqs));
			}

			for (IntegerArray poss : posData) {
				byte encode_poss = 0;

				if (poss.size() > min_encode_len) {
					encode_poss = 1;
				}

				ByteArray b = new ByteArray(poss.size() * Integer.BYTES + 1);
				b.add(encode_poss);

				if (encode_poss == 1) {
					DataCompression.encodeGaps(poss);
					b.addAll(DataCompression.encode(poss));
					// int size1 = poss.size() * Integer.BYTES;
					// int size2 = b.size();
					// double ratio = 1f * size2 / size1;
					// if (ratio < 1) {
					// System.out.printf("poss: %d, ratio=%f (%d/%d)\n",
					// poss.size(), ratio, size2, size1);
					// }
				} else {
					b.addAll(ByteArrayUtils.toByteArray(poss));
				}
				b.trimToSize();

				data.add(b);

				// if(phrs.equals("air_embolization")){
				// System.out.println(pl);
				// System.out.println(new ByteArrayMatrix(data));
				// }
			}
		} else {
			data.add(new ByteArray(phrs.getBytes()));
			data.add(ByteArrayUtils.toByteArray(dseqs.size()));

			{
				IntegerArrayMatrix b = partition(dseqs, max_buf_size);
				data.add(ByteArrayUtils.toByteArray(b.size()));

				for (IntegerArray c : b) {
					data.add(ByteArrayUtils.toByteArray(c));
				}
			}

			{
				for (IntegerArray b : posData) {
					data.add(ByteArrayUtils.toByteArray(b));
				}
			}
		}

		return new ByteArrayMatrix(data);
	}

	public static SPostingList toPostingList(ByteArrayMatrix data, boolean encode) throws Exception {
		String phrs = "";
		IntegerArray dseqs = null;
		IntegerArrayMatrix posData = null;

		if (encode) {
			int i = 0;
			phrs = new String(data.get(i++).values());
			byte encode_seqs = data.get(i++).get(0);
			int pl_size = ByteArrayUtils.toInteger(data.get(i++));

			dseqs = new IntegerArray(pl_size);
			posData = new IntegerArrayMatrix(pl_size);

			if (encode_seqs == 1) {
				dseqs.addAll(DataCompression.decodeToIntegerArray(data.get(i++)));
				DataCompression.decodeGaps(dseqs);
			} else {
				dseqs.addAll(ByteArrayUtils.toIntegerArray(data.get(i++)));
			}

			for (int j = 0; j < dseqs.size(); j++) {
				ByteArray b = data.get(i++);
				byte encode_poss = b.get(0);
				b = b.subArray(1, b.size());
				IntegerArray poss = null;
				if (encode_poss == 1) {
					poss = DataCompression.decodeToIntegerArray(b);
					DataCompression.decodeGaps(poss);
				} else {
					poss = ByteArrayUtils.toIntegerArray(b);
				}
				posData.add(poss);
			}
		} else {
			int i = 0;
			phrs = new String(data.get(i++).values());
			int pl_size = ByteArrayUtils.toInteger(data.get(i++));

			dseqs = new IntegerArray(pl_size);
			posData = new IntegerArrayMatrix(pl_size);

			{
				int size = ByteArrayUtils.toInteger(data.get(i++));
				for (int j = 0; j < size; j++) {
					dseqs.addAll(ByteArrayUtils.toIntegerArray(data.get(i++)));
				}
			}

			{
				for (int j = 0; j < pl_size; j++) {
					posData.add(ByteArrayUtils.toIntegerArray(data.get(i++)));
				}
			}

		}
		return new SPostingList(phrs, dseqs, posData);
	}

	public static long[] writePostingList(SPostingList pl, FileChannel fc, boolean encode) throws Exception {
		return FileUtils.write(toByteArrayMatrix(pl, encode), fc);
	}

	public static void writePostingLists(List<SPostingList> pls, FileChannel fc, boolean encode) throws Exception {
		for (SPostingList pl : pls) {
			FileUtils.write(toByteArrayMatrix(pl, encode), fc);
		}
	}

	private String phrs;

	private IntegerArray dseqs;

	private IntegerArray cnts;

	private IntegerArrayMatrix posData;

	public SPostingList(String phrs, IntegerArray dseqs, IntegerArrayMatrix posData) {
		this.phrs = phrs;
		this.dseqs = dseqs;
		this.posData = posData;

		cnts = new IntegerArray(dseqs.size());

		for (IntegerArray poss : posData) {
			cnts.add(poss.size());
		}
	}

	public IntegerArray getCnts() {
		return cnts;
	}

	public int getCount() {
		return ArrayMath.sum(cnts.values());
	}

	public IntegerArray getDocseqs() {
		return dseqs;
	}

	public String getPhrase() {
		return phrs;
	}

	public IntegerArrayMatrix getPosData() {
		return posData;
	}

	public IntegerArray getPoss(int i) throws Exception {
		return posData.get(i);
	}

	public Posting getPosting(int i) {
		return new Posting(dseqs.get(i), posData.get(i));
	}

	public int size() {
		return dseqs.size();
	}

	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("w=[%s], docs=[%d], cnt=[%d]", phrs, size(), getCount()));
		return sb.toString();
	}

}
