package ohs.corpus.dump;

import ohs.ir.medical.general.MIRPath;

public class DumpHandler {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// {
		// OhsumedDumper dh = new OhsumedDumper(MIRPath.OHSUMED_COL_RAW_DIR, MIRPath.OHSUMED_COL_LINE_DIR);
		// dh.dump();
		// }
		//
		// {
		// ClefEHealthDumper dh = new ClefEHealthDumper(MIRPath.CLEF_EH_2014_COL_RAW_DIR, MIRPath.CLEF_EH_2014_COL_LINE_DIR);
		// dh.dump();
		// }
		//
		{
			TrecGenomicsDumper d = new TrecGenomicsDumper(MIRPath.TREC_GENO_2007_COL_RAW_DIR, MIRPath.TREC_GENO_2007_COL_LINE_DIR);
			d.dump();
		}

		{
			TrecCdsDumper dh = new TrecCdsDumper(MIRPath.TREC_CDS_2014_COL_RAW_DIR, MIRPath.TREC_CDS_2014_COL_LINE_DIR);
			dh.dump();

		}

		{
			TrecCdsDumper dh = new TrecCdsDumper(MIRPath.TREC_CDS_2016_COL_RAW_DIR, MIRPath.TREC_CDS_2016_COL_LINE_DIR);
			dh.dump();
		}

		{
			ClueWebDumper dh = new ClueWebDumper(MIRPath.CLUEWEB_COL_RAW_DIR, MIRPath.CLUEWEB_COL_LINE_DIR);
			dh.dump();
		}

		{
			WikiDumper dh = new WikiDumper(MIRPath.WIKI_COL_XML_FILE, MIRPath.WIKI_COL_LINE_DIR);
			dh.dump();
		}

		System.out.println("process ends.");
	}

}
