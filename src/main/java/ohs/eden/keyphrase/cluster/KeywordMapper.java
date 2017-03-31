package ohs.eden.keyphrase.cluster;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.tree.trie.hash.Node;
import ohs.tree.trie.hash.Trie;
import ohs.tree.trie.hash.Trie.TSResult;
import ohs.tree.trie.hash.Trie.TSResult.MatchType;
import ohs.types.common.StrPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class KeywordMapper {

	public static Trie<Character>[] createDicts(Indexer<StrPair> kwdIndexer) {
		Trie<Character> korDict = new Trie<Character>();
		Trie<Character> engDict = new Trie<Character>();

		for (int i = 0; i < kwdIndexer.size(); i++) {
			int kwdid = i;

			StrPair kwdp = kwdIndexer.getObject(kwdid);

			// for (int i = 0; i < parts.length; i++) {
			// parts[i] = parts[i].substring(1, parts[i].length() - 1);
			// }

			String korKwd = kwdp.getFirst();
			String engKwd = kwdp.getSecond();

			korKwd = KeywordClusterer.normalize(korKwd);
			engKwd = KeywordClusterer.normalize(engKwd);

			if (korKwd.length() > 0) {
				Node<Character> node = korDict.insert(StrUtils.asCharacters(korKwd));
				Set<Integer> data = (Set<Integer>) node.getData();

				if (data == null) {
					data = Generics.newHashSet();
					node.setData(data);
				}

				data.add(kwdid);
			}

			if (engKwd.length() > 0) {
				Node<Character> node = engDict.insert(StrUtils.asCharacters(engKwd));
				Set<Integer> data = (Set<Integer>) node.getData();

				if (data == null) {
					data = Generics.newHashSet();
					node.setData(data);
				}

				data.add(kwdid);
			}
		}

		Trie<Character>[] ret = new Trie[] { korDict, engDict };
		return ret;
	}

	public static Trie<Character>[] createDicts(KeywordData kwdData, int min_kwd_freq) {
		Trie<Character> korDict = new Trie<Character>();
		Trie<Character> engDict = new Trie<Character>();

		for (int i = 0; i < kwdData.getKeywordIndexer().size(); i++) {
			int kwdid = i;

			if (kwdData.getKeywordFreqs().length > 0) {
				int kwd_freq = kwdData.getKeywordFreq(kwdid);
				if (kwd_freq < min_kwd_freq) {
					continue;
				}
			}

			StrPair kwdp = kwdData.getKeywordIndexer().getObject(kwdid);

			// for (int i = 0; i < parts.length; i++) {
			// parts[i] = parts[i].substring(1, parts[i].length() - 1);
			// }

			String korKwd = kwdp.getFirst();
			String engKwd = kwdp.getSecond();

			korKwd = StrUtils.unwrap(korKwd);
			korKwd = KeywordClusterer.normalize(korKwd);
			engKwd = KeywordClusterer.normalize(engKwd);

			if (korKwd.length() > 0) {
				Node<Character> node = korDict.insert(StrUtils.asCharacters(korKwd));
				Set<Integer> data = (Set<Integer>) node.getData();

				if (data == null) {
					data = Generics.newHashSet();
					node.setData(data);
				}

				data.add(kwdid);
			}

			if (engKwd.length() > 0) {
				Node<Character> node = engDict.insert(StrUtils.asCharacters(engKwd));
				Set<Integer> data = (Set<Integer>) node.getData();

				if (data == null) {
					data = Generics.newHashSet();
					node.setData(data);
				}

				data.add(kwdid);
			}
		}

		Trie<Character>[] ret = new Trie[] { korDict, engDict };
		return ret;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		// mapPatents();
		mapOther3P();
		// map3PByJST();
		// map3PByHanlim(true);
		// map3PByHanlim(false);

		System.out.printf("ends.");
	}

	public static final String[] types = { "paper", "report", "patent" };

	public static final String[] attrs = { "docs_no_kwds", "docs_mapped_kwds", "kwds" };

	public static void map3PByHanlim(boolean use_kor_to_eng) throws Exception {
		System.out.println("map 3p by Hanlim Kor to Eng.");

		Indexer<StrPair> kwdIndexer = Generics.newIndexer();
		Map<Integer, Integer> newIdToOldId = Generics.newHashMap();

		String dictFileName = KPPath.KYP_DIR + "hanlim_kor-eng.txt";
		String outputFileName = KPPath.KYP_DIR + "keyword_hanlim-kor-eng_3p.txt.gz";

		if (!use_kor_to_eng) {
			dictFileName = KPPath.KYP_DIR + "hanlim_eng-kor.txt";
			outputFileName = KPPath.KYP_DIR + "keyword_hanlim-eng-kor_3p.txt.gz";
		}

		{
			TextFileReader reader = new TextFileReader(dictFileName, "euc-kr");
			while (reader.hasNext()) {
				String line = reader.next();
				String[] parts = line.split("\t");
				if (parts.length != 3) {
					continue;
				}

				if (reader.getLineCnt() == 1) {
					continue;
				}

				int nid = -1;

				if (use_kor_to_eng) {
					nid = kwdIndexer.getIndex((new StrPair(parts[1], parts[2])));
				} else {
					nid = kwdIndexer.getIndex((new StrPair(parts[2], parts[1])));
				}
				int oid = Integer.parseInt(parts[0]);
				newIdToOldId.put(nid, oid);
			}
			reader.close();
		}

		KeywordMapper kwdMapper = new KeywordMapper();
		kwdMapper.setDicts(createDicts(kwdIndexer));

		DataCollection dc = new DataCollection(KPPath.COL_LINE_DIR);

		String outDir = KPPath.KYP_DIR + "map_halim";

		FileUtils.deleteFilesUnder(outDir);

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (int u = 0; u < types.length; u++) {
			String type = types[u];

			List<File> files = dc.getFiles(type);

			for (int i = 0; i < files.size(); i++) {
				File inFile = files.get(i);
				List<String> docs = FileUtils.readLinesFromText(inFile);

				String outFileName = String.format("%s/%s/%s", outDir, type, inFile.getName());

				// TextFileWriter writer = new TextFileWriter(outFileName);

				for (int j = 0; j < docs.size(); j++) {
					String doc = docs.get(j);
					String[] parts = doc.split("\t");

					parts = StrUtils.unwrap(parts);

					String cn = parts[1];
					String korKwdStr = parts[2];
					String engKwdStr = parts[3];
					String korTitle = parts[4];
					String engTitle = parts[5];
					String korAbs = parts[6];
					String engAbs = parts[7];

					String korContent = korTitle + "\n" + korAbs;
					String engContent = engTitle + "\n" + engAbs;

					korContent = korContent.trim();
					engContent = engContent.trim();

					// if (korTitle.length() == 0 || korAbs.length() == 0 || korKwdStr.length() == 0) {
					// continue;
					// }

					cm.incrementCount(type, attrs[0], 1);

					Counter<Integer> kwdScores1 = kwdMapper.mapKorean(korContent);
					Counter<Integer> kwdScores2 = kwdMapper.mapEnglish(engContent);

					Counter<Integer> kwdScores = Generics.newCounter();
					kwdScores.incrementAll(kwdScores1);
					kwdScores.incrementAll(kwdScores2);

					if (kwdScores.size() > 0) {
						cm.incrementCount(type, attrs[1], 1);
						cm.incrementCount(type, attrs[2], kwdScores.size());

						// writer.write(doc + "\n");

						List<Integer> kwdids = kwdScores.getSortedKeys();

						for (int k = 0; k < kwdids.size(); k++) {
							int kwdid = kwdids.get(k);
							int cnt = (int) kwdScores.getCount(kwdid);
							StrPair kwdp = kwdIndexer.getObject(kwdid);
							int oid = newIdToOldId.get(kwdid);
							// sb.append("\n" + String.format("%s\t%d", kwdp.toString(), kwdid));
							// sb.append("\n" + String.format("%s\t%d", kwdp.toString(), kwdid));

							// writer.write(String.format("%s\t%d\n", cn, oid));

							// writer.write(String.format("%d\t%s\t%d\n", k + 1, kwdp.toString(), cnt));
						}
					}
				}
				// writer.close();
			}
		}

		System.out.println(toString(cm, kwdIndexer.size(), 0));
	}

	public static void map3PByJST() throws Exception {
		System.out.println("map 3P by JST.");

		Indexer<StrPair> kwdIndexer = Generics.newIndexer();
		Map<Integer, Integer> newIdToOldId = Generics.newHashMap();

		{
			TextFileReader reader = new TextFileReader(KPPath.KYP_DIR + "jst_dict.txt", "euc-kr");
			while (reader.hasNext()) {
				String line = reader.next();
				String[] parts = line.split("\t");
				if (parts.length != 3) {
					continue;
				}

				int nid = kwdIndexer.getIndex((new StrPair(parts[1], parts[2])));
				int oid = Integer.parseInt(parts[0]);
				newIdToOldId.put(nid, oid);
			}
			reader.close();

		}

		KeywordMapper kwdMapper = new KeywordMapper();
		kwdMapper.setDicts(createDicts(kwdIndexer));

		DataCollection dc = new DataCollection(KPPath.COL_LINE_DIR);

		String outDir = KPPath.KYP_DIR + "map_jst";

		FileUtils.deleteFilesUnder(outDir);

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (int u = 0; u < types.length; u++) {
			String type = types[u];

			List<File> files = dc.getFiles(type);

			for (int i = 0; i < files.size(); i++) {
				File inFile = files.get(i);
				List<String> docs = FileUtils.readLinesFromText(inFile);

				String outFileName = String.format("%s/%s/%s", outDir, type, inFile.getName());

				// TextFileWriter writer = new TextFileWriter(outFileName);

				for (int j = 0; j < docs.size(); j++) {
					String doc = docs.get(j);
					String[] parts = doc.split("\t");

					parts = StrUtils.unwrap(parts);

					String cn = parts[1];
					String korKwdStr = parts[2];
					String engKwdStr = parts[3];
					String korTitle = parts[4];
					String engTitle = parts[5];
					String korAbs = parts[6];
					String engAbs = parts[7];

					String korContent = korTitle + "\n" + korAbs;
					String engContent = engTitle + "\n" + engAbs;

					korContent = korContent.trim();
					engContent = engContent.trim();

					// if (korTitle.length() == 0 || korAbs.length() == 0 || korKwdStr.length() == 0) {
					// continue;
					// }

					cm.incrementCount(type, attrs[0], 1);

					Counter<Integer> kwdScores1 = kwdMapper.mapKorean(korContent);
					Counter<Integer> kwdScores2 = kwdMapper.mapEnglish(engContent);

					Counter<Integer> kwdScores = Generics.newCounter();
					kwdScores.incrementAll(kwdScores1);
					kwdScores.incrementAll(kwdScores2);

					if (kwdScores.size() > 0) {
						cm.incrementCount(type, attrs[1], 1);
						cm.incrementCount(type, attrs[2], kwdScores.size());

						// writer.write(doc + "\n");

						List<Integer> kwdids = kwdScores.getSortedKeys();

						for (int k = 0; k < kwdids.size(); k++) {
							int kwdid = kwdids.get(k);
							int cnt = (int) kwdScores.getCount(kwdid);
							StrPair kwdp = kwdIndexer.getObject(kwdid);
							int oid = newIdToOldId.get(kwdid);
							// sb.append("\n" + String.format("%s\t%d", kwdp.toString(), kwdid));
							// sb.append("\n" + String.format("%s\t%d", kwdp.toString(), kwdid));

							// writer.write(String.format("%s\t%d\n", cn, oid));

							// writer.write(String.format("%d\t%s\t%d\n", k + 1, kwdp.toString(), cnt));
						}
					}
				}
				// writer.close();
			}
		}

		System.out.println(toString(cm, kwdIndexer.size(), 0));
	}

	public static void mapOther3P() throws Exception {
		System.out.println("map other 3p.");
		KeywordData kwdData = new KeywordData();
		kwdData.readObject(KPPath.KYP_DATA_CLUSTER_SER_FILE);

		Indexer<StrPair> kwdIndexer = kwdData.getKeywordIndexer();

		Set<String> cnSet = Generics.newHashSet();

		for (String cn : kwdData.getDocumentIndxer().getObjects()) {
			cnSet.add(cn);
		}

		KeywordMapper kwdMapper = new KeywordMapper();
		kwdMapper.setDicts(createDicts(kwdIndexer));

		DataCollection dc = new DataCollection(KPPath.COL_LINE_DIR);

		String outDir = KPPath.KYP_DIR + "map_other3p";

		FileUtils.deleteFilesUnder(outDir);

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (int u = 0; u < types.length; u++) {
			String type = types[u];

			List<File> files = dc.getFiles(type);

			for (int i = 0; i < files.size(); i++) {
				File inFile = files.get(i);
				List<String> docs = FileUtils.readLinesFromText(inFile);

				String outFileName = String.format("%s/%s/%s", outDir, type, inFile.getName());

				// TextFileWriter writer = new TextFileWriter(outFileName);

				for (int j = 0; j < docs.size(); j++) {
					String doc = docs.get(j);
					String[] parts = doc.split("\t");

					parts = StrUtils.unwrap(parts);

					String cn = parts[1];
					String korKwdStr = parts[2];
					String engKwdStr = parts[3];
					String korTitle = parts[4];
					String engTitle = parts[5];
					String korAbs = parts[6];
					String engAbs = parts[7];

					String korContent = korTitle + "\n" + korAbs;
					String engContent = engTitle + "\n" + engAbs;

					korContent = korContent.trim();
					engContent = engContent.trim();

					if (cnSet.contains(cn)) {
						continue;
					}

					// if (korTitle.length() == 0 || korAbs.length() == 0 || korKwdStr.length() == 0) {
					// continue;
					// }

					cm.incrementCount(type, attrs[0], 1);

					Counter<Integer> kwdScores1 = kwdMapper.mapKorean(korContent);
					Counter<Integer> kwdScores2 = kwdMapper.mapEnglish(engContent);

					Counter<Integer> kwdScores = Generics.newCounter();
					kwdScores.incrementAll(kwdScores1);
					kwdScores.incrementAll(kwdScores2);

					if (kwdScores.size() > 0) {
						cm.incrementCount(type, attrs[1], 1);
						cm.incrementCount(type, attrs[2], kwdScores.size());

						// writer.write(doc + "\n");

						List<Integer> kwdids = kwdScores.getSortedKeys();

						for (int k = 0; k < kwdids.size(); k++) {
							int kwdid = kwdids.get(k);
							int cnt = (int) kwdScores.getCount(kwdid);
							StrPair kwdp = kwdIndexer.getObject(kwdid);
							// sb.append("\n" + String.format("%s\t%d", kwdp.toString(), kwdid));
							// sb.append("\n" + String.format("%s\t%d", kwdp.toString(), kwdid));

							// writer.write(String.format("%s\t%d\n", cn, oid));

							// writer.write(String.format("%d\t%s\t%d\n", k + 1, kwdp.toString(), cnt));
						}
					}
				}
				// writer.close();
			}
		}

		System.out.println(toString(cm, kwdIndexer.size(), 0));
	}

	public static String toString(CounterMap<String, String> cm, int kwd_size, long doc_size) {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("kwd_size:\t%d\n", kwd_size));
		sb.append(String.format("doc_size:\t%d\n", doc_size));

		sb.append("<Type>");
		for (int i = 0; i < types.length; i++) {
			sb.append("\t" + types[i]);
		}

		sb.append("\n");

		for (int i = 0; i < attrs.length; i++) {
			sb.append(attrs[i]);

			for (int j = 0; j < types.length; j++) {
				int cnt = (int) cm.getCount(types[j], attrs[i]);
				sb.append("\t" + cnt);
			}
			sb.append("\n");
		}

		sb.append("avg kwds");

		for (int j = 0; j < types.length; j++) {
			int docs = (int) cm.getCount(types[j], attrs[1]);
			int kwds = (int) cm.getCount(types[j], attrs[2]);
			double avg_kwds = 1f * kwds / docs;
			sb.append("\t" + avg_kwds);
		}

		return sb.toString();
	}

	private Trie<Character> korDict;

	private Trie<Character> engDict;

	public KeywordMapper() {

	}

	public Counter<Integer> map(String text, boolean is_korean) {
		Character[] cs = StrUtils.asCharacters(KeywordClusterer.normalize(text));

		Counter<Integer> ret = Generics.newCounter();

		Trie<Character> dict = is_korean ? korDict : engDict;

		for (int i = 0; i < cs.length; i++) {
			for (int j = i + 1; j < cs.length; j++) {
				TSResult<Character> sr = dict.search(cs, i, j);

				if (sr.getMatchType() == MatchType.EXACT_KEYS_WITH_DATA) {
					// if (is_korean) {
					// Set<Integer> set = (Set<Integer>) sr.getMatchNode().getData();
					// for (int kwdid : set) {
					// ret.incrementCount(kwdid, 1);
					// }
					// } else {
					String next_ch = " ";
					String prev_ch = " ";

					if (j < cs.length) {
						next_ch = cs[j].toString();
					}

					if (i - 1 >= 0) {
						prev_ch = cs[i - 1].toString();
					}

					if ((next_ch.equals(" ")) && prev_ch.equals(" ")) {
						Set<Integer> set = (Set<Integer>) sr.getMatchNode().getData();
						//
						// StringBuffer sb = new StringBuffer();
						//
						// int start = CommonMath.min(i, i - 3, 0);
						// int end = CommonMath.max(j, j + 3, cs.length);
						//
						// for (int k = start; k < end; k++) {
						// sb.append(cs[k].toString());
						// }
						//
						// System.out.printf("-> %s\n", sb.toString());
						for (int kwdid : set) {
							ret.incrementCount(kwdid, 1);

							// System.out.println(kwdData.getKeywordIndexer().getObject(kwdid));
						}
					}
					// }

				} else if (sr.getMatchType() == MatchType.EXACT_KEYS_WITHOUT_DATA) {

				} else {
					break;
				}
			}
		}

		return ret;
	}

	public Counter<Integer> mapEnglish(String text) {
		return map(text, false);

		// if (kwdScores.size() == 0) {
		// return kwdScores;
		// }
		//
		// SetMap<Integer, Integer> clusterToKwds = Generics.newSetMap();
		//
		// for (int kwdid : kwdScores.keySet()) {
		// // StrPair kwdp = kwdData.getKeywordIndexer().getObject(kwdid);
		// clusterToKwds.put(kwdToCluster.get(kwdid), kwdid);
		// }
		//
		// if (clusterToKwds.size() == 1) {
		// return kwdScores;
		// }
		//
		// SparseVector input = VectorUtils.toSparseVector(patentWordCnts, wordIndexer, false);
		//
		// VectorMath.unitVector(input);
		//
		// Counter<Integer> catCosines = Generics.newCounter();
		//
		// for (int cid : clusterToKwds.keySet()) {
		// SparseVector cent = cents.get(cid);
		// if (cent == null) {
		// continue;
		// }
		//
		// double cosine = VectorMath.cosine(input, cent, false);
		// catCosines.setCount(cid, cosine);
		// }
		//
		// if (catCosines.size() > 1) {
		// List<Integer> cids = catCosines.getSortedKeys();
		//
		// kwdids = Generics.newHashSet();
		//
		// double cutoff = 0.5;
		//
		// while (kwdids.isEmpty() && cutoff >= 0) {
		//
		// for (int i = 0; i < cids.size(); i++) {
		// int cid = cids.get(i);
		// double cosine = catCosines.getCount(cid);
		//
		// if (cosine < cutoff) {
		// break;
		// }
		//
		// for (int kwdid : clusterToKwds.get(cid)) {
		// kwdids.add(kwdid);
		// // StrPair kwdp = kwdData.getKeywordIndexer().getObject(kwdid);
		// // System.out.println(kwdp);
		// }
		// }
		//
		// cutoff -= 0.1;
		//
		// }
		//
		// }
		//
		// return kwdids;
	}

	public Counter<Integer> mapKorean(String text) {
		return map(text, true);
	}

	public Set<Integer> searchKorean(String text) {
		text = KeywordClusterer.normalize(text);

		Character[] cs = StrUtils.asCharacters(text);
		Set<Integer> kwdids = Generics.newHashSet();
		for (int start = 0; start < cs.length; start++) {
			TSResult<Character> sr = korDict.find(cs, start);

			if (sr.getMatchType() == MatchType.FAIL) {
				start++;
			} else {
				int end = sr.getMatchLoc() + 1;

				StringBuffer sb = new StringBuffer();

				for (int j = start; j < end; j++) {
					sb.append(cs[j].charValue());
				}

				String s = sb.toString();

				Set<Integer> set = (Set<Integer>) sr.getMatchNode().getData();

				kwdids.addAll(set);

				start = end - 1;
			}
		}
		return kwdids;
	}

	public void setDicts(Trie<Character>[] dicts) {
		korDict = dicts[0];
		engDict = dicts[1];
	}

	public void setEngDict(Trie<Character> korDict) {
		this.korDict = korDict;
	}

	public void setKorDict(Trie<Character> korDict) {
		this.korDict = korDict;
	}

}
