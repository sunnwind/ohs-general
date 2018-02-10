package ohs.fake;

import java.io.File;
import java.util.List;

import ohs.corpus.type.DocumentCollection;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.search.app.WordSearcher;
import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LDocumentCollection;
import ohs.nlp.ling.types.LToken;
import ohs.types.generic.Counter;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class SynonymFinder {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		SynonymFinder sf = new SynonymFinder();
		sf.find();

		System.out.println("process ends.");
	}

	private LDocumentCollection readData() throws Exception {
		LDocumentCollection ret = new LDocumentCollection();
		for (File file : FileUtils.getFilesUnder(FNPath.DATA_DIR + "data_pos")) {
			String name = file.getName();
			if (name.startsWith("M1_train-1") || name.startsWith("M1_train-2") || name.startsWith("M1_train-3")) {
				for (String line : FileUtils.readLinesFromText(file)) {
					List<String> ps = StrUtils.unwrap(StrUtils.split("\t", line));

					String id = ps.get(0);
					String label = ps.get(1);
					label = label.equals("0") ? "non-fake" : "fake";

					LDocument d = LDocument.newDocument(ps.get(2));
					d.getAttrMap().put("id", id);
					d.getAttrMap().put("label", label);

					if (ps.get(3).length() > 0) {
						d.getAttrMap().put("cor_title", ps.get(3));
					}
					ret.add(d);
				}
			}
		}
		return ret;
	}

	public void find() throws Exception {
		Vocab vocab = DocumentCollection.readVocab(FNPath.NAVER_NEWS_COL_DC_DIR + "vocab.ser");
		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(FNPath.NAVER_DATA_DIR + "/emb/glove_ra.ser");

		WordSearcher ws = new WordSearcher(vocab, E, null);

		LDocumentCollection C = readData();

		List<LToken> ts = C.getTokens();

		Counter<String> wordCnts = Generics.newCounter();

		for (LToken t : ts) {
			String word = t.getString(0);
			String pos = t.getString(1);

			if (word.length() < 2) {
				continue;
			}

			if (pos.startsWith("N") || pos.startsWith("V")) {
				wordCnts.incrementCount(word, 1);
			}
		}

		List<String> words = wordCnts.getSortedKeys();

		for (int i = 0; i < words.size(); i++) {
			String word = words.get(i);
			Counter<String> c = ws.getSimilarWords(word, 5);
			System.out.println(c.toString());

		}
	}

}
