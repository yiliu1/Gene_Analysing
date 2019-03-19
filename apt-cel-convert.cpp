00001 ////////////////////////////////////////////////////////////////
00002 //
00003 // Copyright (C) 2005 Affymetrix, Inc.
00004 //
00005 // This library is free software; you can redistribute it and/or modify
00006 // it under the terms of the GNU Lesser General Public License 
00007 // (version 2.1) as published by the Free Software Foundation.
00008 // 
00009 // This library is distributed in the hope that it will be useful, but
00010 // WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
00011 // or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
00012 // for more details.
00013 // 
00014 // You should have received a copy of the GNU Lesser General Public License
00015 // along with this library; if not, write to the Free Software Foundation, Inc.,
00016 // 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
00017 //
00018 ////////////////////////////////////////////////////////////////
00019 
00020 
00021 //
00022 #include "calvin_files/converters/cel/src/CELFileConversionOptions.h"
00023 #include "calvin_files/converters/cel/src/CELFileConverter.h"
00024 #include "file/TsvFile/TsvFile.h"
00025 #include "util/AptVersionInfo.h"
00026 #include "util/Err.h"
00027 #include "util/Fs.h"
00028 #include "util/Guid.h"
00029 #include "util/LogStream.h"
00030 #include "util/PgOptions.h"
00031 #include "util/Util.h"
00032 #include "util/Verbose.h"
00033 //
00034 #include <cstdlib>
00035 #include <cstring>
00036 #include <fstream>
00037 #include <iostream>
00038 #include <map>
00039 #include <string>
00040 #include <vector>
00041 
00042 #ifndef WIN32
00043 #include <unistd.h>
00044 #endif
00045 
00046 
00047 using namespace std;             // Who isn't?
00048 using namespace affymetrix_cel_converter;
00049 using namespace affx;
00050 using namespace affxutil;
00051 
00052 class AOptions {
00053     public:
00054       bool help;
00055       int verbose;
00056       string outDir;
00057       bool inPlace;
00058       CELFileVersionType format;
00059       string progName;
00060       string version;
00061       string execGuid;
00062       string timeStr;
00063       string commandLine;
00064       string chipType;
00065       bool setDatName;
00066       int verbosity;
00067       vector<string> celFiles;
00068 };
00069 
00070 void define_aptcelconvert_options(PgOptions* opts)
00071 {
00072   opts->setUsage("apt-cel-convert - program to convert cel files to different types.\n"
00073                  "usage:\n"
00074                  "   apt-cel-convert -f text -o text-cels *.CEL\n"
00075                  "   \n"
00076                  "   apt-cel-convert --format xda --out-dir text-cels --cel-files cel-files.txt\n");
00077 
00078   opts->defineOption("h", "help", PgOpt::BOOL_OPT,
00079                      "Display program options and extra documentation about possible analyses.",
00080                      "false");
00081  opts->defineOption("i", "in-place", PgOpt::BOOL_OPT,
00082                     "Convert the file in place. Over-write existing file.",
00083                     "false");
00084  opts->defineOption("f", "format", PgOpt::STRING_OPT,
00085                     "Set the output cel file format type. Valid values: xda, text, agcc.",
00086                     "");
00087  opts->defineOption("", "log-file", PgOpt::STRING_OPT,
00088                     "The output log file. Defaults to location of output with name apt-cel-convert.log.",
00089                     "");
00090  opts->defineOption("v", "verbose", PgOpt::INT_OPT,
00091                     "How verbose to be with status messages 0 - quiet, 1 - usual messages, 2 - more messages.",
00092                     "1");
00093  opts->defineOption("", "version", PgOpt::BOOL_OPT,
00094                     "Display version information.",
00095                     "false");
00096  opts->defineOption("", "set-dat-name", PgOpt::BOOL_OPT,
00097                     "Set the DAT file name to match that of the cel file name.",
00098                     "false");
00099  opts->defineOption("", "cel-files", PgOpt::STRING_OPT,
00100                     "Text file specifying cel files to process, one per line with the first line being 'cel_files'.",
00101                     "");
00102  opts->defineOption("o", "out-dir", PgOpt::STRING_OPT,
00103                     "Directory to write result files into.",
00104                     "");
00105  opts->defineOption("", "chip-type", PgOpt::STRING_OPT,
00106                     "Force the new cel file to be this chip type.",
00107                     "");
00108 }
00109 
00110 /** Fill in options given command line arguments. */
00111 void fillInOptions(PgOptions *opts, AOptions &o, int argc) {
00112   if(opts->getBool("help") || argc == 1) 
00113     o.help = true;
00114   else
00115     o.help = false;
00116 
00117   o.progName = opts->getProgName();
00118   o.progName = Fs::basename(o.progName);
00119   o.version = AptVersionInfo::versionToReport();
00120   o.execGuid = affxutil::Guid::GenerateNewGuid();
00121 
00122   //
00123   o.format = Unknown_Version;
00124   if (opts->get("format")=="text")
00125     o.format = GCOS_Version3;
00126   else if(opts->get("format")=="xda")
00127     o.format = GCOS_Version4;
00128   else if (opts->get("format")=="agcc")
00129     o.format = Calvin_Version1;
00130   else if (opts->get("format")!="")
00131     Err::errAbort("Invalid cel format: " + opts->get("format"));
00132 
00133   o.verbosity = opts->getInt("verbose");
00134   o.outDir = opts->get("out-dir");
00135   o.chipType = opts->get("chip-type");
00136   o.setDatName = opts->getBool("set-dat-name");
00137   o.inPlace = opts->getBool("in-place");
00138   
00139   /* Read in cel file list from other file if specified. */
00140   if (opts->get("cel-files")!="") {
00141     affx::TsvFile tsv;
00142 #ifdef WIN32
00143     tsv.m_optEscapeOk = false;
00144 #endif
00145     std::string celFiles = opts->get("cel-files");
00146     if(tsv.open(celFiles) != TSV_OK) {
00147       Err::errAbort("Couldn't open cell-files file: " + celFiles);
00148     }
00149     std::string file;
00150     tsv.bind(0, "cel_files", &file, TSV_BIND_REQUIRED);
00151     tsv.rewind();
00152     while(tsv.nextLevel(0) == TSV_OK) {
00153       o.celFiles.push_back(Util::cloneString(file.c_str()));
00154     }
00155     tsv.close();
00156     Verbose::out(1, "Read " + ToStr(o.celFiles.size()) + " cel files from: " + Fs::basename(celFiles));
00157   }
00158   else {
00159     for(int i = 0; i < opts->getArgCount(); i++) 
00160       o.celFiles.push_back(opts->getArg(i));
00161   }
00162 
00163   /* post processing. */
00164   o.timeStr = Util::getTimeStamp();
00165 
00166   if(!o.help) {
00167     /* Some sanity checks. */
00168     if(o.format == Unknown_Version) 
00169       Err::errAbort("Must provide a valid cel file format with --format option.");
00170     if(o.celFiles.empty()) 
00171       Err::errAbort("Must specify at least one cel file to convert.");
00172     if(Util::sameString(o.outDir.c_str(), "") && !o.inPlace)
00173       Err::errAbort("Must specify an output directory (--out-dir) or in place conversion (--in-place)");
00174     if(!Util::sameString(o.outDir.c_str(), "") && o.inPlace)
00175       Err::errAbort("Must specify an output directory (--out-dir) OR in place conversion (--in-place). Not both.");
00176 
00177     if(!o.inPlace) {
00178         if(!Fs::isWriteableDir(o.outDir.c_str())) {
00179             if(Fs::mkdirPath(o.outDir) != APT_OK) {
00180                 Err::errAbort("Can't make or write to directory: " + ToStr(o.outDir));
00181             }
00182         }
00183     }
00184   }
00185 }
00186 
00187 /** 
00188  * Report out some basics about what we think the run is:
00189  * 
00190  * @param o - Program options.
00191  */
00192 void reportBasics(AOptions &o, const string &version) {
00193   Verbose::out(3, "version=" + version);
00194 #ifndef WIN32
00195   char name[8192];
00196   if(gethostname(name, ArraySize(name)) == 0) 
00197     Verbose::out(3, "host=" + ToStr(name));
00198   else 
00199     Verbose::out(3, "host=unknown");
00200   if(getcwd(name, ArraySize(name)) != NULL) 
00201     Verbose::out(3, "cwd=" + ToStr(name));
00202   else
00203     Verbose::out(3, "cwd=unknown");
00204 #endif /* WIN32 */
00205   Verbose::out(3, "command-line=" + o.commandLine);
00206 }
00207 
00208 /** Everybody's favorite function... */
00209 int main(int argc, char *argv[]) {
00210   try {
00211     const string version ("NON-OFFICIAL-RELEASE");
00212     ofstream logOut;
00213     string logName;
00214     PgOptions *opts = NULL;
00215     unsigned int i = 0;
00216     AOptions ourOpts;
00217     
00218     /* Parse options. */
00219     opts = new PgOptions();
00220     define_aptcelconvert_options(opts);
00221     opts->parseArgv(argv);
00222     /* Need to check for the version option before fillInOptions
00223         to avoid conflicts with those consistency checks */
00224     if(opts->getBool("version")) {
00225         cout << "version: " << version << endl;
00226         exit(0);
00227     }
00228     
00229     fillInOptions(opts, ourOpts, argc);
00230     ourOpts.commandLine = opts->commandLine();
00231     
00232     Verbose::setLevel(ourOpts.verbosity);
00233     // Do we need help? (I know I do...)
00234     if(ourOpts.help) {
00235         opts->usage();
00236         cout << "version: " << version << endl;
00237         exit(0);
00238     }
00239     else {
00240         time_t startTime = time(NULL);
00241     
00242         /* Set up the logging and message handlers. */
00243         if (opts->get("log-file") != "") {
00244             logName = opts->get("log-file");
00245         }
00246         else if (ourOpts.inPlace) {
00247           logName = Fs::join(".","apt-cel-convert.log");
00248         }
00249         else {
00250           logName = Fs::join(ourOpts.outDir,"apt-cel-convert.log");
00251         }
00252         Fs::mustOpenToWrite(logOut, logName.c_str());
00253         LogStream log(3, &logOut);
00254         Verbose::pushMsgHandler(&log);
00255         Verbose::pushProgressHandler(&log);
00256         Verbose::pushWarnHandler(&log);
00257     
00258         reportBasics(ourOpts, version);
00259     
00260         /* Do the heavy lifting */
00261             CELFileConverter converter;
00262         for(i = 0; i < ourOpts.celFiles.size(); i++) {
00263     
00264             /* Figure out extra conversion options */
00265             CELFileConversionOptions convertOpts;
00266             // Copy (ie no format change) will fail if options are supplied -- so only provide
00267             // them if we need to
00268             CELFileConversionOptions *optsPtr = NULL;
00269             if(ourOpts.chipType != "") {
00270                 convertOpts.m_ChipType = Util::cloneString(ourOpts.chipType.c_str());
00271                 optsPtr = &convertOpts;
00272             }
00273             if(ourOpts.setDatName) {
00274                 string newName = Fs::basename(ourOpts.celFiles[i]);
00275                 newName = newName.substr(0,newName.rfind("."));
00276                 convertOpts.m_DATFileName = Util::cloneString(newName.c_str());
00277                 optsPtr = &convertOpts;
00278             }
00279     
00280             if(ourOpts.inPlace) {
00281                 Verbose::out(1, "Converting " + ToStr(ourOpts.celFiles[i]));
00282                     if (converter.ConvertFile( ourOpts.celFiles[i].c_str(), ourOpts.format, optsPtr ) == false)
00283                     Err::errAbort("Could not convert cel file " + ToStr(ourOpts.celFiles[i]) + ": " + 
00284                                 CELFileConverterErrorMessage(converter.ErrorCode()));
00285             } 
00286             else {
00287               string outfile = Fs::join(ourOpts.outDir,Fs::basename(ourOpts.celFiles[i]));
00288     
00289               Verbose::out(1, "Converting " + ToStr(ourOpts.celFiles[i]) + " to " + outfile);
00290     
00291                     if (converter.ConvertFile( ourOpts.celFiles[i].c_str(), outfile.c_str(), ourOpts.format, optsPtr ) == false)
00292                     Err::errAbort("Could not convert cel file " + ToStr(ourOpts.celFiles[i]) + ": " + 
00293                                 CELFileConverterErrorMessage(converter.ErrorCode()));
00294             }
00295     
00296             if(convertOpts.m_DATFileName != NULL)
00297                 delete[] convertOpts.m_DATFileName;
00298             if(convertOpts.m_ChipType != NULL)
00299                 delete[] convertOpts.m_ChipType;
00300         }
00301     
00302         /* Close out log(s) */
00303         time_t endTime = time(NULL);
00304         int t = int(  (float)(endTime - startTime) / 60.0 * 100); // convert to minutes
00305         Verbose::out(1, ToStr("Run took approximately: ") + ToStr((float)t/100) + ToStr(((float)t/100) > 1 ? " minutes." : " minute."));
00306         if(!ourOpts.inPlace) 
00307             logOut.close();
00308     }
00309     
00310     delete opts;
00311     return 0;
00312   } 
00313   catch(...) {
00314       Verbose::out(1,"Unexpected Error: uncaught exception.");
00315       return 1;
00316   }
00317   return 1;
00318 }
00319 