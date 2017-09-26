package ohs.ir.search.index;

import java.util.List;
import java.util.Map;

import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;
import ohs.utils.Timer;

public class MemoryInvertedIndex extends InvertedIndex {

	private Map<Integer, PostingList> plm;

	public MemoryInvertedIndex(Map<Integer, PostingList> plm, int doc_cnt, Vocab vocab) {
		this.plm = plm;
		this.vocab = vocab;
		this.doc_cnt = doc_cnt;
	}

	public MemoryInvertedIndex copyShallow() throws Exception {
		return new MemoryInvertedIndex(plm, doc_cnt, vocab);
	}

	public PostingList getPostingList(int w) throws Exception {
		Timer timer = Timer.newTimer();
		PostingList ret = plm.get(w);
		boolean is_cached = false;

		if (print_log) {
			String word = vocab == null ? "null" : vocab.getObject(w);
			System.out.printf("PL: cached=[%s], word=[%s], %s, time=[%s]\n", is_cached, word, is_cached, ret,
					timer.stop());
		}

		if (ret == null) {
			ret = new PostingList(-1, new IntegerArray(), new IntegerMatrix());
		}

		return ret;
	}

	public List<PostingList> getPostingLists(int i, int j) throws Exception {
		int size = j - i;
		List<PostingList> ret = Generics.newArrayList(size);
		for (int k = i; k < j; k++) {
			ret.add(getPostingList(k));
		}
		return ret;
	}

	public int size() {
		return plm.size();
	}

}
