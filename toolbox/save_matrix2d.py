import numpy as np
import pdb

def save_matrix2d_to_file(matrix, save_path):
    '''assert(~isempty(matrix), 'The matrix to save is empty while saving to file.');
    fileID = get_fileID_for_saving(save_path);
    [nrows, ncols] = size(matrix);

    for i = 1:nrows
        for j = 1:ncols
            fprintf(fileID, '%05.5f ', matrix(i, j));     
        end
        fprintf(fileID, '\n');
    end
    
    fclose(fileID);
    return nrows, ncols'''
    [rows, cols] = matrix.shape
    np.savetxt(save_path, matrix, fmt='%4.2f')
    return rows, cols

def parse_matrix_file(file_path, debug_mode=False):
    data = np.loadtxt(file_path)
    nrows = data.shape[0]

    return data, nrows


if __name__ == "__main__":
    pass