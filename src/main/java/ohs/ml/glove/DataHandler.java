package ohs.ml.glove;

import ohs.ir.medical.general.MIRPath;

public class DataHandler {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DataHandler dh = new DataHandler();
		dh.convert();

		System.out.println("process ends.");
	}

	public void convert() throws Exception {
		String[] dirs = { MIRPath.OHSUMED_COL_DIR, MIRPath.CLEF_EH_2014_COL_DIR, MIRPath.TREC_GENO_2007_COL_DIR,
				MIRPath.TREC_CDS_2014_COL_DIR, MIRPath.TREC_CDS_2016_COL_DIR, MIRPath.WIKI_COL_DIR };

		// for (int u = 0; u < dirs.length; u++) {
		// String dir = dirs[u];
		// String scDir = dir + "dc/";
		// String ccDir = dir + "cocnt/";
		// String modelFileName = new File(dir).getParent() + "/glove_model.ser.gz";
		// String outputFileName = new File(dir).getParent() + "/glove_model.txt.gz";
		//
		// DocumentCollection lsc = new DocumentCollection(dir + "dc/");
		//
		// GloveModel M = new GloveModel();
		// M.readObject(new File(dir).getParent() + "/glove_model.ser.gz");
		//
		// WordVectorModel vecs = new WordVectorModel(lsc.getVocab(), M.getAveragedModel());
		//
		// List<String> words = Generics.newArrayList(lsc.getVocab());
		//
		// Collections.sort(words);
		//
		// TextFileWriter writer = new TextFileWriter(outputFileName);
		// writer.write(String.format("vocab_size:\t%d", lsc.getVocab().size()));
		// writer.write(String.format("\nhidden_size:\t%d\n", M.getW1().colSize()));
		//
		// for (int i = 0; i < words.size(); i++) {
		// String word = words.get(i);
		// DenseVector v = vecs.getVector(word);
		//
		// StringBuffer sb = new StringBuffer();
		// sb.append(word);
		//
		// for (int j = 0; j < v.size(); j++) {
		// sb.append(String.format("\t%s", v.value(j) + ""));
		// }
		//
		// writer.write(sb.toString());
		//
		// if (i != words.size() - 1) {
		// writer.write("\n");
		// }
		// }
		//
		// writer.close();
		// }
	}

}
