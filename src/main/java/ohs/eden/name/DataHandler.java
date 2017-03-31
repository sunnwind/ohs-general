package ohs.eden.name;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.VectorUtils;
import ohs.matrix.SparseVector;
import ohs.ml.svm.wrapper.LibLinearTrainer;
import ohs.ml.svm.wrapper.LibLinearWrapper;
import ohs.string.search.ppss.Gram;
import ohs.string.search.ppss.GramGenerator;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class DataHandler {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DataHandler dh = new DataHandler();
		dh.makeNameData();
		// dh.makeTestData();

		System.out.println("process ends.");
	}

	public void makeTestData() throws Exception {
		String[] fileNames = { NamePath.NAME_PER_KOR_FILE, NamePath.ORG_FILE };

		GramGenerator korGG = new GramGenerator(3);
		GramGenerator engGG = new GramGenerator(4);

		Indexer<String> featIndexer = Generics.newIndexer();
		Indexer<String> labelIndexer = Generics.newIndexer();

		labelIndexer.add("Person");
		labelIndexer.add("Organization");

		List<SparseVector> data = Generics.newArrayList();
		List<Integer> labels = Generics.newArrayList();

		Counter<Integer> ccc = Generics.newCounter();

		for (int i = 0, id = 0; i < fileNames.length; i++) {
			Counter<String> c = FileUtils.readStringCounterFromText(fileNames[i]);

			List<String> keys = c.getSortedKeys();

			for (int j = 0; j < keys.size(); j++) {
				String[] two = keys.get(j).split("\t");
				two = StrUtils.unwrap(two);

				String kor = two[0];
				String eng = two[1];

				Counter<String> cc = Generics.newCounter();

				if (i == 0) {
					if (kor.length() > 0 && kor.length() <= 5) {

						if (kor.endsWith("회") || kor.endsWith("터") || kor.endsWith("회의") || kor.endsWith("합")) {
							continue;
						} else {
							for (Gram g : korGG.generateQGrams(kor.toLowerCase())) {
								cc.incrementCount(g.getString(), 1);
							}
						}

						if (eng.length() > 0) {
							// cc.incrementCount("#kor_eng_len#", eng.length());
							for (Gram g : engGG.generateQGrams(eng.toLowerCase())) {
								cc.incrementCount(g.getString(), 1);
							}
						}
					}

				} else if (i == 1) {
					if (kor.length() > 0) {
						// cc.incrementCount("#kor_char_len#", kor.length());
						for (Gram g : korGG.generateQGrams(kor.toLowerCase())) {
							cc.incrementCount(g.getString(), 1);
						}
					}

					if (eng.length() > 0) {
						// cc.incrementCount("#kor_eng_len#", eng.length());
						for (Gram g : engGG.generateQGrams(eng.toLowerCase())) {
							cc.incrementCount(g.getString(), 1);
						}
					}
				}

				if (cc.size() > 0) {
					SparseVector sv = VectorUtils.toSparseVector(cc, featIndexer, true);
					data.add(sv);

					labels.add(i);

					ccc.incrementCount(i, 1);
				}
			}
		}

		System.out.println(ccc.toString());

		TermWeighting.tfidf(data);

		List<Integer> locs = Generics.newArrayList();

		for (int i = 0; i < data.size(); i++) {
			locs.add(i);
		}

		List<Integer>[] twoSets = DataSplitter.splitInOrder(locs, ArrayMath.array(0.8, 0.2));

		List<SparseVector> trainData = Generics.newArrayList();
		List<SparseVector> testData = Generics.newArrayList();
		List<Integer> trainLabels = Generics.newArrayList();
		List<Integer> testLabels = Generics.newArrayList();

		for (int loc : twoSets[0]) {
			trainData.add(data.get(loc));
			trainLabels.add(labels.get(loc));
		}

		for (int loc : twoSets[1]) {
			testData.add(data.get(loc));
			testLabels.add(labels.get(loc));
		}

		LibLinearTrainer t = new LibLinearTrainer();
		LibLinearWrapper wrapper = t.train(labelIndexer, featIndexer, trainData, trainLabels);
		wrapper.write(NamePath.SVM_MODEL_FILE);

		String res = wrapper.evalute(testData, testLabels);

		System.out.println(res);
	}

	public void makeNameData() throws Exception {
		Counter<String> korPerNames = Generics.newCounter();
		Counter<String> korOrgNames = Generics.newCounter();
		Counter<String> engPerNames = Generics.newCounter();
		Counter<String> engOrgNames = Generics.newCounter();

		{
			Map<String, String> map1 = Generics.newHashMap();
			Map<String, String> map2 = Generics.newHashMap();

			for (String line : FileUtils.readLinesFromText(NamePath.PATENT_MAP_1_FILE)) {
				String[] parts = line.split("\t");
				String APAGT_CD = parts[0];
				String INDV_CORP_TPCD = parts[1];
				map1.put(APAGT_CD, INDV_CORP_TPCD);
			}

			TextFileReader reader = new TextFileReader(NamePath.PATENT_FILE);
			reader.setPrintNexts(false);

			List<String> labels = Generics.newArrayList();

			while (reader.hasNext()) {
				reader.printProgress();

				String line = reader.next();

				String[] parts = line.split("\t");

				if (reader.getLineCnt() == 1) {
					labels = Arrays.asList(parts);
				} else {

					if (parts.length != labels.size()) {
						continue;
					}

					parts = StrUtils.unwrap(parts);

					String pak = parts[0];
					String pae = parts[1];
					String pac = parts[2];
					String cn = parts[3];

					List<String> kors = getItems(pak);
					List<String> engs = getItems(pae);
					List<String> codes = getItems(pac);

					if ((kors.size() > 0 || engs.size() > 0) && codes.size() > 0) {
						String kor = kors.size() > 0 ? kors.get(0) : "";
						String eng = engs.size() > 0 ? engs.get(0) : "";
						String code = codes.get(0);
						String newCode = map1.get(code);

						String[] s = new String[] { kor, eng };
						String res = StrUtils.join("\t", StrUtils.wrap(s));

						if (newCode.equals("A0904") || newCode.equals("A0906") || newCode.equals("A0908")) {
							if (newCode.equals("A0906")) {
								engPerNames.incrementCount(res, 1);
							} else {
								korPerNames.incrementCount(res, 1);
							}
						} else {
							if (newCode.equals("A0905")) {
								engOrgNames.incrementCount(res, 1);
							} else {
								korOrgNames.incrementCount(res, 1);
							}
						}
					}
				}
			}
			reader.printProgress();
			reader.close();
		}

		{
			TextFileReader reader = new TextFileReader(NamePath.PAPER_FILE);
			reader.setPrintNexts(false);

			List<String> labels = Generics.newArrayList();

			while (reader.hasNext()) {
				reader.printProgress();

				String line = reader.next();

				String[] parts = line.split("\t");

				if (reader.getLineCnt() == 1) {
					labels = Arrays.asList(parts);
				} else {

					if (parts.length != labels.size()) {
						continue;
					}

					parts = StrUtils.unwrap(parts);

					String cn = parts[0];
					String auk = parts[1];
					String aue = parts[2];
					String csk = parts[3];
					String cse = parts[4];

					List<String> korAuths = getItems(auk);
					List<String> engAuths = getItems(aue);

					List<String> korOrgs = getItems(csk);
					List<String> engOrgs = getItems(cse);

					if (korAuths.size() > 0 || engAuths.size() > 0) {
						if (korAuths.size() == engAuths.size()) {
							for (int i = 0; i < korAuths.size(); i++) {
								String kor = korAuths.get(i);
								String eng = engAuths.get(i);
								String[] s = new String[] { kor, eng };
								korPerNames.incrementCount(StrUtils.join("\t", StrUtils.wrap(s)), 1);
							}
						} else {
							if (korAuths.size() > engAuths.size()) {
								for (int i = 0; i < korAuths.size(); i++) {
									String kor = korAuths.get(i);
									String[] s = new String[] { kor, "" };
									korPerNames.incrementCount(StrUtils.join("\t", StrUtils.wrap(s)), 1);
								}
							} else {
								for (int i = 0; i < engAuths.size(); i++) {
									String eng = engAuths.get(i);
									String[] s = new String[] { "", eng };
									korPerNames.incrementCount(StrUtils.join("\t", StrUtils.wrap(s)), 1);
								}
							}
						}
					}

					if (korOrgs.size() > 0 || engOrgs.size() > 0) {
						if (korOrgs.size() == engOrgs.size()) {
							for (int i = 0; i < korOrgs.size(); i++) {
								String kor = korOrgs.get(i);
								String eng = engOrgs.get(i);
								String[] s = new String[] { kor, eng };
								korOrgNames.incrementCount(StrUtils.join("\t", StrUtils.wrap(s)), 1);
							}
						} else {
							if (korOrgs.size() > engOrgs.size()) {
								for (int i = 0; i < korOrgs.size(); i++) {
									String kor = korOrgs.get(i);
									String[] s = new String[] { kor, "" };
									korOrgNames.incrementCount(StrUtils.join("\t", StrUtils.wrap(s)), 1);
								}
							} else {
								for (int i = 0; i < engOrgs.size(); i++) {
									String eng = engOrgs.get(i);
									String[] s = new String[] { "", eng };
									korOrgNames.incrementCount(StrUtils.join("\t", StrUtils.wrap(s)), 1);
								}
							}
						}
					}
				}
			}
			reader.printProgress();
			reader.close();
		}

		FileUtils.writeStringCounterAsText(NamePath.NAME_PER_KOR_FILE, korPerNames);
		FileUtils.writeStringCounterAsText(NamePath.NAME_PER_ENG_FILE, engPerNames);
		FileUtils.writeStringCounterAsText(NamePath.NAME_ORG_KOR_FILE, korOrgNames);
		FileUtils.writeStringCounterAsText(NamePath.NAME_ORG_ENG_FILE, engOrgNames);
	}

	private List<String> getItems(String s) {
		List<String> ret = Generics.newArrayList();
		for (String item : StrUtils.split(";", s)) {
			if (item.length() > 0) {
				ret.add(item);
			}
		}
		return ret;
	}

}
