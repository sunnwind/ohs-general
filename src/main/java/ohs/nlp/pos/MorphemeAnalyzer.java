package ohs.nlp.pos;

import java.util.List;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.TokenAttr;
import ohs.tree.trie.hash.HMTNode;
import ohs.tree.trie.hash.HMTrie;
import ohs.tree.trie.hash.HMTrie.TSResult;
import ohs.tree.trie.hash.HMTrie.TSResult.MatchType;
import ohs.types.generic.SetMap;
import ohs.types.generic.Triple;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class MorphemeAnalyzer {

	public static void main(String[] args) throws Exception {
		System.out.println("proces begins.");

		TextTokenizer t = new TextTokenizer();
		MorphemeAnalyzer a = new MorphemeAnalyzer();

		{
			String document = "프랑스랑은 세계적인 의상 디자이너 엠마누엘 웅가로가 실내 장식용 직물 디자이너로 나섰다.\n";
			// String document = "우승은 프랑스일테니까!\n";

			MDocument doc = t.tokenize(document);
			a.analyze(doc);
		}

		// SejongReader r = new SejongReader(NLPPath.POS_DATA_FILE,
		// NLPPath.POS_TAG_SET_FILE);
		// while (r.hasNext()) {
		// MDocument doc = r.next();
		//
		// StringBuffer sb = new StringBuffer();
		//
		// for (int i = 0; i < doc.size(); i++) {
		// MTSentence sent = doc.getSentence(i);
		// MultiToken[] mts = MultiToken.toMultiTokens(sent.getTokens());
		//
		// for (int j = 0; j < mts.length; j++) {
		// sb.append(mts[j].getValue(TokenAttr.WORD));
		// if (j != mts.length - 1) {
		// sb.append(" ");
		// }
		// }
		// if (i != doc.size() - 1) {
		// sb.append("\n");
		// }
		// }
		//
		// MDocument newDoc = t.tokenize(sb.toString());
		// a.analyze(newDoc);
		//
		// }
		// r.close();

		System.out.println("proces ends.");
	}

	private HMTrie<Character> wordDict;

	private HMTrie<Character> suffixDict;

	private HMTrie<SJTag> conRules;

	private SetMap<String, String> analDict;

	public MorphemeAnalyzer() throws Exception {
		readDicts();
	}

	public void analyze(MDocument doc) {
		for (int i = 0; i < doc.size(); i++) {
			MSentence sent = doc.getSentence(i);
			MultiToken[] mts = sent.toMultiTokens();

			for (int j = 0; j < mts.length; j++) {
				MultiToken mt = mts[j];
				String wp = mt.get(TokenAttr.WORD);

				Set<String> res = analDict.get(wp, false);

				if (res == null) {
					analyze(wp);
				} else {

				}

				// Set<String> morphemes = analDict.get(eojeol, false);

				// if (morphemes == null) {

				// } else {
				// String s = StrUtils.join(" # ", morphemes);
				// mt.setValue(TokenAttr.POS, s);
				// }
			}
		}

	}

	public Set<String> analyze(String wp) {
		Set<String> ret = Generics.newHashSet();

		Character[] cs = StrUtils.asCharacters(wp);
		int len = cs.length;

		List<Triple<Integer, Integer, Set<String>>> list1 = Generics.newArrayList();
		List<Triple<Integer, Integer, Set<String>>> list2 = Generics.newArrayList();

		{

			int start = 0;

			while (start < cs.length) {
				TSResult<Character> sr = suffixDict.find(cs, start);
				if (sr.getMatchType() == MatchType.FAIL) {
					start++;
				} else {
					int end = sr.getMatchLoc() + 1;
					Set<String> set = (Set<String>) sr.getMatchNode().getData();
					list2.add(Generics.newTriple(start, end, set));
					start = end;
				}
			}

			for (int i = 0; i < list2.size(); i++) {
				Triple<Integer, Integer, Set<String>> t = list2.get(i);
				System.out.printf("%s, %s\n", wp.substring(t.getFirst(), t.getSecond()), t.getThird());
			}
			System.out.println();
		}

		{
			int start = 0;

			while (start < cs.length) {
				TSResult<Character> sr = wordDict.find(cs, start);
				if (sr.getMatchType() == MatchType.FAIL) {
					start++;
				} else {
					int end = sr.getMatchLoc() + 1;
					Set<String> set = (Set<String>) sr.getMatchNode().getData();
					list1.add(Generics.newTriple(start, end, set));
					start = end;
				}
			}

			for (int i = 0; i < list1.size(); i++) {
				Triple<Integer, Integer, Set<String>> t = list1.get(i);
				System.out.printf("%s, %s\n", wp.substring(t.getFirst(), t.getSecond()), t.getThird());
			}
			System.out.println();
		}

		return ret;
	}

	public String getString(List<Character> s, int start, int end) {
		StringBuffer sb = new StringBuffer();
		for (int i = start; i < end; i++) {
			sb.append(s.get(i).charValue());
		}
		return sb.toString();
	}

	public void readDicts() throws Exception {
		analDict = Generics.newSetMap();

		conRules = HMTrie.newTrie();

		// for (String line : FileUtils.readLines(NLPPath.DICT_ANALYZED_FILE)) {
		// String[] parts = line.split("\t");
		//
		// for (int i = 1; i < parts.length; i++) {
		// analDict.put(parts[0], parts[1]);
		//
		// String[] subParts = parts[i].split(" \\+ ");
		// SJTag[] poss = new SJTag[subParts.length];
		//
		// for (int j = 0; j < subParts.length; j++) {
		// String[] two = subParts[j].split(" / ");
		// poss[j] = SJTag.valueOf(two[1]);
		// }
		//
		// conRules.insert(poss);
		// }
		// }

		wordDict = HMTrie.newTrie();

		for (String line : FileUtils.readLinesFromText(NLPPath.DICT_WORD_FILE)) {
			String[] parts = line.split("\t");
			Character[] cs = StrUtils.asCharacters(parts[0]);
			HMTNode<Character> node = wordDict.insert(cs);

			Set<String> set = (Set<String>) node.getData();

			if (set == null) {
				set = Generics.newHashSet(parts.length - 1);
				node.setData(set);
			}

			for (int i = 1; i < parts.length; i++) {
				set.add(parts[i]);
			}
		}

		suffixDict = HMTrie.newTrie();

		for (String line : FileUtils.readLinesFromText(NLPPath.DICT_SUFFIX_FILE)) {
			String[] parts = line.split("\t");

			String suffix = parts[0];

			if (suffix.startsWith("~")) {
				suffix = suffix.substring(1);
			}

			Character[] cs = StrUtils.asCharacters(suffix);
			HMTNode<Character> node = suffixDict.insert(cs);

			Set<String> set = (Set<String>) node.getData();

			if (set == null) {
				set = Generics.newHashSet(parts.length - 1);
				node.setData(set);
			}

			for (int i = 1; i < parts.length; i++) {
				String[] toks = parts[i].split(MultiToken.DELIM.replace("+", "\\+"));
				set.add(StrUtils.join(MultiToken.DELIM, toks, 1, toks.length));
			}
		}
	}

}
