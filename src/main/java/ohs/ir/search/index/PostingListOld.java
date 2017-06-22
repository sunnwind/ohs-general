package ohs.ir.search.index;

import java.nio.channels.FileChannel;
import java.util.List;

import ohs.corpus.type.DataCompression;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.FileUtils;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.ByteSize;
import ohs.utils.Generics;

public class PostingListOld {

	public static IntegerArrayMatrix partition(IntegerArray a, int chunk_size) {
		int window_size = chunk_size / Integer.BYTES;
		List<IntegerArray> ret = Generics.newArrayList(sizeOfPartitions(a, chunk_size));
		int s = 0;
		while (s < a.size()) {
			int e = Math.min(a.size(), s + window_size);
			ret.add(a.subArray(s, e));
			s = e;
		}
		return new IntegerArrayMatrix(ret);
	}

	public static PostingListOld readPostingList(FileChannel fc, boolean encode) throws Exception {
		PostingListOld ret = null;
		if (fc.position() < fc.size()) {
			ret = toPostingList(FileUtils.readByteArrayMatrix(fc), encode);
		}
		return ret;
	}

	public static List<PostingListOld> readPostingLists(FileChannel fc, boolean encode) throws Exception {
		List<PostingListOld> ret = Generics.newLinkedList();
		if (fc.position() < fc.size()) {
			ret.add(toPostingList(FileUtils.readByteArrayMatrix(fc), encode));
		}
		return ret;
	}

	public static int sizeOfByteBuffer(PostingListOld pl) {
		int ret = Integer.BYTES;
		ret += Integer.BYTES * pl.getDocSeqs().size() * 2;
		ret += Integer.BYTES * pl.getPosData().sizeOfEntries();
		ret += Integer.BYTES * pl.getPosData().size();
		return ret;
	}

	public static int sizeOfPartitions(IntegerArray a, int chunk_size) {
		int window_size = chunk_size / Integer.BYTES;
		int s = 0;
		int ret = 0;
		while (s < a.size()) {
			int e = Math.min(a.size(), s + window_size);
			s = e;
			ret++;
		}
		return ret;
	}

	public static ByteArrayMatrix toByteArrayMatrix(PostingListOld pl, boolean encode) throws Exception {
		int w = pl.getWord();
		IntegerArray dseqs = pl.getDocSeqs();
		IntegerArrayMatrix posData = pl.getPosData();

		int min_encode_len = 20;

		int max_buf_size = FileUtils.DEFAULT_BUF_SIZE;

		List<ByteArray> data = null;

		{
			int size = 10;
			size += pl.getPosData().size();
			size += sizeOfPartitions(pl.getDocSeqs(), max_buf_size) + 1;
			data = Generics.newArrayList(size);
		}

		if (encode) {
			byte encode_dseqs = 0;

			if (dseqs.size() > min_encode_len) {
				encode_dseqs = 1;
			}

			data.add(ByteArrayUtils.toByteArray(w));
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
			}
		} else {
			data.add(ByteArrayUtils.toByteArray(w));
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

	public static PostingListOld toPostingList(ByteArrayMatrix data, boolean encode) throws Exception {
		int w = 0;
		IntegerArray dseqs = null;
		IntegerArrayMatrix posData = null;

		if (encode) {
			int i = 0;
			w = ByteArrayUtils.toInteger(data.get(i++));
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
			w = ByteArrayUtils.toInteger(data.get(i++));
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

		return new PostingListOld(w, dseqs, posData);
	}

	public static long[] writePostingList(PostingListOld pl, FileChannel fc, boolean encode) throws Exception {
		return FileUtils.write(toByteArrayMatrix(pl, encode), fc);
	}

	private int w;

	private IntegerArray dseqs;

	private IntegerArray cnts;

	private IntegerArrayMatrix posData;

	private IntegerArrayMatrix endPosData;

	private int total_cnt;

	public PostingListOld(int w, IntegerArray dseqs, IntegerArray cnts, IntegerArrayMatrix posData) {
		this.w = w;
		this.dseqs = dseqs;
		this.cnts = cnts;
		this.posData = posData;
	}

	public PostingListOld(int w, IntegerArray dseqs, IntegerArrayMatrix posData) {
		this.w = w;
		this.dseqs = dseqs;
		this.posData = posData;
		cnts = new IntegerArray(dseqs.size());
		for (IntegerArray poss : posData) {
			cnts.add(poss.size());
			total_cnt += poss.size();
		}
	}

	public int getCount() {
		return total_cnt;
	}

	public IntegerArray getCounts() {
		return cnts;
	}

	public IntegerArray getDocSeqs() {
		return dseqs;
	}

	public IntegerArrayMatrix getEndPosData() {
		return endPosData;
	}

	public IntegerArrayMatrix getPosData() {
		return posData;
	}

	public IntegerArray getPoss(int i) throws Exception {
		return posData.get(i);
	}

	public Posting getPosting(int i) {
		return new Posting(dseqs.get(i), posData.get(i), endPosData == null ? null : endPosData.get(i));
	}

	public int getWord() {
		return w;
	}

	public void setEndPosData(IntegerArrayMatrix starts) {
		this.endPosData = starts;
	}

	public void setPosData(IntegerArrayMatrix posData) {
		this.posData = posData;
	}

	public void setWord(int w) {
		this.w = w;
	}

	public int size() {
		return dseqs.size();
	}

	public String toString() {

		long bytes = ByteArrayUtils.sizeOfByteBuffer(dseqs) * 2 + ByteArrayUtils.sizeOfByteBuffer(posData);
		ByteSize bs = new ByteSize(bytes);
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("w=[%d], docs=[%d], cnt=[%d], mem=%s", w, size(), getCount(), bs));
		return sb.toString();
	}

}
