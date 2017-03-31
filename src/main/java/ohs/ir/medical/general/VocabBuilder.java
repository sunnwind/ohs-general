package ohs.ir.medical.general;

import org.apache.lucene.index.Fields;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.util.BytesRef;

import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.ir.lucene.common.CommonFieldNames;
import ohs.types.generic.Counter;

public class VocabBuilder {

	public static void build(String indexDirName, String outputFileName) throws Exception {
		IndexSearcher is = SearcherUtils.getIndexSearcher(indexDirName);

		Fields fields = MultiFields.getFields(is.getIndexReader());

		Terms t = fields.terms(CommonFieldNames.CONTENT);

		long term_size = t.size();
		System.out.println(t.getSumTotalTermFreq());

		TermsEnum te = t.iterator();

		BytesRef bytesRef = null;
		PostingsEnum pe = null;

		// Indexer<String> wordIndexer = Generics.newIndexer();
		// SparseVector wordCnts = new SparseVector();
		// SparseVector docFreqs = new SparseVector();

		long num_words = 0;

		TextFileWriter writer = new TextFileWriter(outputFileName);
		writer.write(String.format("###DOC SIZE###\t%d", is.getIndexReader().maxDoc()));

		while ((bytesRef = te.next()) != null) {
			pe = te.postings(pe, PostingsEnum.ALL);
			pe.nextDoc();
			// if (pe.nextDoc() != 0) {
			// throw new AssertionError();
			// }

			if (++num_words % 100000 == 0) {
				System.out.printf("\r[%d words]", num_words);
			}

			long cnt = te.totalTermFreq();
			int doc_freq = te.docFreq();
			String word = bytesRef.utf8ToString();
			// int w = wordIndexer.getIndex(word);
			// wordCnts.add((int) cnt);
			// docFreqs.add(doc_freq);

			writer.write(String.format("\n%s\t%d\t%d", word, cnt, doc_freq));
		}
		writer.close();
		System.out.printf("\r[%d words]\n", num_words);

		// FileUtils.writeStrCounter(outPathName, wordCnts);

		// Vocab vocab = new Vocab(wordIndexer, wordCnts.getValues(),
		// docFreqs.getValues(), is.getIndexReader().maxDoc());
		// vocab.write(outPathName);
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		for (int i = 4; i < MIRPath.IndexDirNames.length; i++) {
			buildFromTempFile(MIRPath.IndexDirNames[i], MIRPath.VocFileNames[i]);
		}

		// merge(MIRPath.VocFileNames, MIRPath.VOCAB_FILE);

		// for (int i = 0; i < MIRPath.TaskDirs.length; i++) {
		// String outPathName = MIRPath.TaskDirs[i] + "vocab2.txt";
		// build2(MIRPath.IndexDirNames[i], outPathName);
		// }

		System.out.println("process ends.");
	}

	public static void merge(String[] inputFileNames, String outputFileName) throws Exception {
		Counter<String> c = new Counter<String>();

		for (String inputFileName : inputFileNames) {
			c.incrementAll(FileUtils.readStringCounterFromText(inputFileName));
		}

		FileUtils.writeStringCounterAsText(outputFileName, c);
	}

}
