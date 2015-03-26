import fiona

def fix_shapefile(infile_path, outfile_path):
    
    #WRITES TO A NEW SHAPEFILE WITHOUT NULL ENTRIES

    with fiona.open(infile_path, 'r') as infile:
        with fiona.open(outfile_path, 'w', **infile.meta) as outfile:
            for rec in range(len(infile)):
	        if infile[rec]['geometry'] != None:
		    outfile.write(infile[rec])
