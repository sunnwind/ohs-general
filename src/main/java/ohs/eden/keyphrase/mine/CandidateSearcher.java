package ohs.eden.keyphrase.mine;

import java.util.List;

import it.unimi.dsi.fastutil.doubles.Double2IntOpenCustomHashMap;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;
import ohs.tree.trie.hash.Node;
import ohs.tree.trie.hash.Trie;
import ohs.types.generic.Counter;
import ohs.types.generic.Pair;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class CandidateSearcher {

	private Trie<String> patTrie;

	private Counter<String> patCnts;

	public CandidateSearcher(Counter<String> patCnts) {
		this.patCnts = patCnts;
		patTrie = new Trie<String>();

		for (String posPat : patCnts.getSortedKeys()) {
			List<String> words = StrUtils.split(posPat);
			Node<String> node = patTrie.insert(words);
			node.setFlag(true);
		}
		patTrie.trimToSize();
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Counter<String> patCnts = FileUtils.readStringCounterFromText(KPPath.KYP_DIR + "ext/kwd_pat.txt");

		CandidateSearcher cs = new CandidateSearcher(patCnts);

		for (String line : FileUtils.readLinesFromText(KPPath.KYP_DIR + "ext/label_data.txt")) {
			String[] ps = line.split("\t");
			ps = StrUtils.unwrap(ps);

			String kwdStr = ps[0];
			String title = ps[1];
			String abs = ps[2];

			MDocument doc = MDocument.newDocument(title + "\n" + abs.replace(StrUtils.LINE_REP, "\n"));
			List<List<Pair<Integer, Integer>>> posData = cs.search(doc);
		}

		System.out.println("process ends.");
	}

	public List<List<Pair<Integer, Integer>>> search(MDocument doc) {
		List<List<Pair<Integer, Integer>>> ret = Generics.newArrayList(doc.size());

		for (int i = 0; i < doc.size(); i++) {
			MSentence sent = doc.get(i);
			List<Token> ts = sent.getTokens();
			List<Pair<Integer, Integer>> poss = Generics.newArrayList();

			int j = 0;
			while (j < ts.size()) {
				int k = j;

				Node<String> node = patTrie.getRoot();

				while (k < ts.size()) {
					Token t = ts.get(k);
					String pos = t.get(TokenAttr.POS);

					if (node.hasChild(pos)) {
						node = node.getChild(pos);
						if (node.getFlag()) {
							poss.add(Generics.newPair(j, k + 1));
						}
					} else {
						break;
					}
					k++;
				}

				if (j == k) {
					j++;
				} else {
					j = k;
				}
			}
			ret.add(poss);
		}
		return ret;
	}

}
