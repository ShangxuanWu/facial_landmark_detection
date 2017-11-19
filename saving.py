# Author: Xinshuo Weng and Shangxuan Wu
# email: {xinshuo.weng, wushx6}@gmail.com

# this function convert a struct to string (only applied to the field which can be converted to a string) and save it to file
import toolbox.util
import pdb

def save_struct(struct_save, save_path, debug_mode):
    '''if not debug_mode:
        debug_mode = True;
    

    if debug_mode:
    	assert toolbox.util.isstring(save_path), 'The input path should be a string to a file.'
    	assert isstruct(struct_save), 'The stuff to save should be a struct.'
    '''
    with open(save_path, 'w') as f:
        
        for k, v in struct_save.__dict__.iteritems():
            print(k)
            print(v)
            f.write('%s    %s\n' % (k, v))

	'''file_id = get_fileID_for_saving(save_path);
    nrows = 0;

    # convert struct to string and write to file
    fields = fieldnames(struct_save);
    for field_index = 1:length(fields)
        fields_tmp = fields{field_index};
        value = getfield(struct_save, fields_tmp);

        if isstruct(value)
            continue;
        elif isscalar(value)
            try
                str_tmp = num2str(value);
            catch
                str_tmp = 'failed to convert to string!!!';
                if debug_mode
                    fprintf('field %s cannot converted to string\n', fields_tmp);
                
            
        elif ismatrix(value)
            try
                str_tmp = mat2str(value);
            catch
                str_tmp = 'failed to convert to string!!!';
                if debug_mode
                    fprintf('field %s cannot converted to string\n', fields_tmp);
                
                        
        elif ischar(value)
            str_tmp = value;
        else
            continue;

        # write the key and value string
        fprintf(file_id, '%-30s\t\t\t\t', fields_tmp); 
        fprintf(file_id, '%s\n', str_tmp); 
        nrows = nrows + 1;
    
    fclose(file_id);
    return nrows'''

    return
