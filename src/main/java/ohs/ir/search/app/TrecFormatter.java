package ohs.ir.search.app;

import java.util.Collections;
import java.util.List;

import ohs.io.FileUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class TrecFormatter {

	public static void format(CounterMap<String, String> sr, String runName, String outputFileName) throws Exception {
		List<String> qids = Generics.newArrayList(sr.keySet());
		Collections.sort(qids);

		List<String> res = Generics.newArrayList(sr.totalSize());

		for (int i = 0; i < qids.size(); i++) {
			String qid = qids.get(i);
			Counter<String> scores = sr.getCounter(qid);
			List<String> dids = scores.getSortedKeys();

			for (int j = 0; j < dids.size(); j++) {
				String did = dids.get(j);
				double score = scores.getCount(did);
				int rank = j + 1;
				res.add(String.format("%s\tQ0\t%s\t%d\t%f\t%s", qid, did, rank, score, runName));
			}
		}

		if (outputFileName != null) {
			FileUtils.writeAsText(outputFileName, StrUtils.join("\n", res));
		}
	}

}
