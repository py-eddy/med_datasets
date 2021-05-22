CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�KƧ       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�V/   max       P�ɧ       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��Q�   max       =8Q�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?^�Q�   max       @F��\(��       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @vt�����       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P�           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�3            7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       =�P       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4��       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4��       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�Sn       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�   max       C�O�       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          [       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          G       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�V/   max       P��/       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�@N���U   max       ?տH˒:       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��Q�   max       =8Q�       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?^�Q�   max       @F��\(��       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vt            P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P�           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @���           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D^   max         D^       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?ye+��a   max       ?տH˒:     0  ^         "                     G   '         [   '      '      /                        (            
   Q      	   0         '                  3   
               $               /         G                                       N�}�NM%O�?4N�+EN�mNIR�NX�O��Ni_�P�}xPjgO(�NgP�P�d�ON��IO�v�M�ԷP�NG�+N`�wNE��N�IO�~O���NP��P>vO
�
M݀�O&eN�4P�ɧN�QO�iO�&�N�:O�>�P�PN��oOT�O	iOK��N:�O���N�m�Nz��N�6KN�DmN�ߝPG2N�l�O~��N;�;O>O���O��O\n�O�x�M�V/NEc�O�2N���O�ȵOeELOm5TN�zNd�N`0&O&:Ol�On<�=8Q�<o;o;o;o���
�o�o�t��t��t��T����C���1��j���ͼ�/��/��`B��`B��`B��`B���o�o�o�o�o�o�o�+�\)�\)����w�49X�49X�8Q�8Q�8Q�8Q�<j�D���D���D���H�9�H�9�L�ͽP�`�P�`�Y��ixսixսm�h�m�h�q���q���q���u�u�u�y�#�y�#�y�#�����t���t���t����T���T��Q�!#/<HOMHB<9/)#!!!!!!*/<HSMH<9/**********���$/22-)���������������������������������������������QT]akmypmha`TPQQQQQQ������������������06CLSWVSHC6*���x����������{xxxxxxxx�#<{������{b<0
��������������������������.+)�����������������������������
0Ubt{wbUI0����=B[_mw����th[OB<686=oty������������~utooETmz���������ziaTGAE@BIOQPOBA?@@@@@@@@@@�������������������������	���������������������������)*.)�����������������������������������������
#0<@87
�����#/2;9/#!cnz������������xmjcc').6BBHNLIB6-)&" ''MO[_[[[^[ONJMMMMMMMMahnt�������������tha����������������������((#$45..#�����������

��������55?BENOSQQZNB53/.05546AOY[hkrrvwtxth[JB4@BCO[[\[[YOHB;@@@@@@����������������$BNgt��������t[B5)#$������������������������������������������������

������+25>BNWZYYNB5)%��������������������#/<HRbijl_UH/#knpz~��������zuqnmkkSU^ajnwpnba`UJSSSSSS

#02<9210#


����������������������������������������DIPTbn{�������{bUP?Drt��������}tsnrrrrrr������������������������������������������������������������$&/<HU^ijicaYUH<8/#$QUamzz}��~zpaTPMKKQ]fmz~~|||zqmjebZYYZ]����)B=5����������������������������������������������S[gt������������tf[S[gt�����}trga][[[[[[����������������������������������������st��������������{tps))5A=75)��������������������������������������������
��������������

 
������������!'286)�����ŹŷůűŭŹ��������������źŹŹŹŹŹŹ���������������������������������������غ�ֺɺú��ºɺֺ�������������U�L�H�=�<�H�U�a�f�k�f�a�U�U�U�U�U�U�U�U�����������������������������������������t�o�h�c�[�W�[�h�k�tĀāćā�t�t�t�t�t�t��������%�$������������.��	���ݾ����	��.�3�8�>�C�H�G�;�.�����������������������������������j�W�D�/�(�7�Z�s���������������������y�`�G�8�7�?�G�`�y���ѿҿǿ��������y�z�q�n�a�_�^�g�k�n�z�{ÁÇÓ×ÜàÓÇ�z�N�M�I�L�N�Z�a�g�p�k�g�Z�N�N�N�N�N�N�N�N�ӽ�������t�z�����Ľݾ����޽���ӿT�I�%�.�;�T�m�y���������������������m�T�`�Z�T�M�K�T�`�i�m�w�y�������}�y�m�i�`�`�������������������
��3�:�.�#� ��
����l�b�l�r�x���������x�l�l�l�l�l�l�l�l�l�l�b�Q�K�`�{ŔŠ������������������ŠŇ�n�b�5�0�(�(�(�4�5�A�J�N�R�O�N�E�A�5�5�5�5�5������������&�$�������������������¦°ª¦�f�_�]�f�r�����������r�f�f�f�f�f�f�f�f�n�k�a�U�U�a�n�n�zÇÈÓàáìàÓÇ�z�nìÓËÆ�z�pÇáìù�����������������ì�H�C�F�H�U�a�k�l�a�U�H�H�H�H�H�H�H�H�H�H��������������"�/�H�g�t�t�l�H�;������׾������������������ʾѾҾʾž��������������������������ûĻĻû����������������������������x�u�v�w�x�}�������������������f�_�Z�M�A�?�A�T�Z�f�s�t�����������s�f�!�����:�G�y���ݾ�(�s�{�s�9���Ľ��S�!����������������������������������������ŠşŔőŔşŠŭŹ��������������ŹŭŠŠ����Ϲ����������ùܹ�����%�)�)���������������������������������������������ʼ����������Լڼ��!�3�4������ּʾ�ʾžƾɾ˾׾����	� �%�#���������=�8�=�=�I�V�b�n�o�o�o�b�V�I�=�=�=�=�=�=�e�`�Y�Y�`�e�m�r�~�����������������~�r�e�ɺ��������������������ɺֺߺ���ֺպ��T�S�H�;�8�;�>�Q�T�a�m�z������|�z�m�a�T��������	���(�(��������������������A�;�5�6�;�A�M�Z�f�s���������s�f�Z�M�A������x��������������������������������Ŀ¿��������ĿѿԿݿݿݿؿѿĿĿĿĿĿĻ����ܻ׻лʻϻлܻ�������������������������Ŀѿݿ߿ݿٿѿĿ�������������ƎƈƁ�zƁƎƚƧƳƽ������ƳƧƚƎƎƎƎ�û����_�G�>�@�J�_�x�ûܻ�����
��ܻ��C�@�:�C�N�O�Q�\�h�m�h�_�\�O�C�C�C�C�C�Cā�}�u�h�[�O�B�@�C�W�[�h�tĎēĔĒēčā��������������������
���������m�l�a�X�W�[�a�l�m�v�z�������������{�z�mE�E�E�E�E�E�E�E�E�E�E�E�E�E�FF
E�E�E�E����������������������	�����	���������A�5�-�+�5�A�s�����������������s�g�Z�N�A�[�O�E�I�S�f�tĒĥĳĶĹĹĶĮĦĚā�t�[�����������������������������������������	�����$�'�+�&�$������������������	�����.�5�A�D�;�7�)��ĦěĜĤĦĳĿ����������ĿĳĦĦĦĦĦĦŹŭŚŔŇ�~ŇŎŔťŭŹ��������������Ź�0�(�#����#�&�0�<�I�U�b�m�y�e�U�I�<�0²¯¦²¿����������������¿²�z�x�n�m�n�n�p�zÇÓÕÙÓÓÇ�|�z�z�z�z�x�n�t�x�����������������x�x�x�x�x�x�x�x���������ûлܻ��ܻлû���������������������ݿڿݿ�����(�5�9�A�I�A�5�(���g�]�g�g�s�s�����������������������s�p�g��������'�4�@�F�H�M�O�N�H�@�4�'� N � ! + = � E h = ? X W @ l 1 8 I W j { T g < ` J 3 L = J b g } 4 I K i { N 4 B # &  - : @ 4 9 z R > C ` V 6 E ~ b ~ K N /   3 R 7 J S  ^ S    �  z    �  0  �  N  �  �  �  W  �  z  �     �  0    �  �  �  p  �  `  �  f  �  A     �    	X  �  6  �  x  �  �  �  6  .  �  �  M  �  s  �  �  �  �  �    �  W  �    �  @  B  {  Q  �    �    �  |    -  U  &=�P;o�o�49X��o�t��49X�ě��T�����
�@����ͼ�1��S��}������o���P�o�+�+��P�0 Žy�#�#�
��t��0 Žt��<j�,1��h�'@��� Ž<j������1�T���m�h��C��u�P�`�ȴ9�ixսu�]/�y�#�aG���-�q������y�#��t�������1���P����%���������C���1������{�� Ž������j�\��FB�/B�fBdcB ��B4��A���B�B/��B
��B&�<B*ڋB� B�VB&�Bl3B
~NA���Bn�BL�B�BhBJ�B�B=�B>VBvB�RB�"B�5BrB!��B��Bz,B��BB�B��B-�UB	�'BrSB"�B#��ByB�B�dBy�BELB%4AB��B ��B(Z'B
(B0zB�aB�rB-�A�#_A�W�BO�B��B�B
v3B	��B��B(B
��B��B�B��B�B�pB�BB�B=�B�dB �pB4��A��B 6B0?�B
\�B&LB*�fB@<BkOB' /B<�B
\�A���B@]B�+B�
B}�B@jB�WB�RBt-B�B��B��B³B?�B!�*B@�BC�B�BC&BD�B-�HB	��B�B!פB#�B�B{B�uB�\B?�B%ShB�2B ��B(?GB
DB��B��B��B�A���A��>B@sB��B.�B	��B	��BAfBAVB
N�B�@B��B��B�B��B҈A�еA�}L@J��A�U1AK�1A�	
?�CA\�hA/�hA��AsŬA��A�{pA%�uAmiAjY�A���@��%A�p�A�2�A�!3A�iW@�@}A�֖A·YAŵ.A�L~AL� @���@��tAA�1A#�
A���A�k�?��A��AzAX=SB��@9j@3G�A�z�A�@XA?�-AIO�Ay��@�KAx��B��@���BeA�QNBPXA��C�SnA��EA��!A���B�aB	K�A�	A�{oA�+�A��A���A��@�`�@��&A���A�(b@��*A��TA���@H	�A�_nAL-�A�i3?���A[@A/
�A���Ao��A�}�A��wA&�An�Aj�^A�^@�A��cA��%A�z}A���@�;RA�4�A���A�z�A�AK��@��H@��A@0�AUA�N�A�{?�A�{4A04AX�&B�6?��@3��A�sAA��A>�AIC�Az�$@��/Aw�B?�@�ƯB?�A�}BD�A��.C�O�A��@A�сA�rLB�GB	}�A� PA���A���A���A�� A�j�@���@�ZqA���A�b@��R   	      #                     H   '         [   (      '      /                     	   )            
   R      
   0          (                  3   
               %               /         H                                                               !      ;   ?         C   %      !      )                  #      /               O         %      +   '                                    1               !         %                                                               !      7   5         )         !      '                        /               G         !                                             /                                                               N�}�NM%OS�NL�1N�mN��NX�O��Ni_�P��)P9|�O(�NgP�O�h�O��N��IO�v�M�ԷO��NG�+N`�wNE��N�IO�~Oz%HNP��P>vN���M݀�O ��N�4P��/N�QO�iO�>N�:O2�O:�N��oOT�N��OK��N:�O���N�m�Nz��N�6KN�DmN�ߝP1gRN�l�O~��N;�;N��sO�`kO��O\n�O�u�M�V/NEc�O�2N���Oa;jOeELOٟN�zNd�N`0&O&:N칤On<�  �  @  b  D  �  A  
    x  �  A  �  �  �    1  l  �  5    U  �  �  M  �  �  r  J  R    <  �  �  4  �    �  �  ;  �  I  �  �  �  �  �  a  =  �  c    d  �  '  �  A  �  F    �  �  �  k  V  A  �  G  /  �  %  �=8Q�<o�#�
��o;o�ě��o�o�t��T���e`B�T����C��aG���`B���ͼ�/��/�+��`B��`B��`B���o�#�
�o�o�+�o�C��+�8Q�\)���,1�49X�ixս�+�8Q�8Q�H�9�<j�D���P�`�D���H�9�H�9�L�ͽP�`�]/�Y��ixսixսq���y�#�q���q����\)�u�u�u�y�#��%�y�#��O߽�t���t���t����T���Q�!#/<HOMHB<9/)#!!!!!!*/<HSMH<9/**********'),+)( �������������������������������������������QT^almqnmfa[TRQQQQQQ������������������06CLSWVSHC6*���x����������{xxxxxxxx#<b������{b<0
��������������������������.+)��������������������������0IUbpqlbUI<0#
��BO[ht�����zth[OA;;9Boty������������~utooETmz���������ziaTGAE@BIOQPOBA?@@@@@@@@@@�������������������������	���������������������������)*.)������������������������������������������
#/8/*"
����#/2;9/#!cnz������������xmjcc")6ABGLKGB6/)'$ """"MO[_[[[^[ONJMMMMMMMMeht��������������the������������������������#(-+(���������

��������55?BENOSQQZNB53/.055BOT[hqqtvurvtk[TC68B@BCO[[\[[YOHB;@@@@@@����
��������V[cgt��������tg_[XVV����������������������������������������������

�������+25>BNWZYYNB5)%�������������������� #/<HP`aegaUH/# knpz~��������zuqnmkkSU^ajnwpnba`UJSSSSSS

#02<9210#


����������������������������������������KRWbn{��������{bUCDKrt��������}tsnrrrrrr������������������������������������������������������������/<HU[afgg`UQH</*$%'/QUamzz}��~zpaTPMKKQ]fmz~~|||zqmjebZYYZ]��")1)�����������������������������������������������S[gt������������tf[S[gt�����}trga][[[[[[����������������������������������������stv�������������|wts))5A=75)��������������������������������������������
��������������

������������!'286)�����ŹŷůűŭŹ��������������źŹŹŹŹŹŹ���������������������������������������غֺպ˺ʺֺۺ�������
��������ֺ��U�P�H�@�@�H�U�_�a�e�c�a�U�U�U�U�U�U�U�U�����������������������������������������t�q�h�e�[�Z�[�h�h�t�}āĄā�t�t�t�t�t�t��������%�$������������.��	���ݾ����	��.�3�8�>�C�H�G�;�.���������������������������������m�Z�H�:�2�C�Z�������������������������y�`�G�=�=�G�T�`�m�������ɿ�����꿫�y�z�q�n�a�_�^�g�k�n�z�{ÁÇÓ×ÜàÓÇ�z�N�M�I�L�N�Z�a�g�p�k�g�Z�N�N�N�N�N�N�N�N�����������������Ľݽ������ѽƽĽȽĽ��R�C�;�=�U�m�y�����������������������m�R�`�Z�T�M�K�T�`�i�m�w�y�������}�y�m�i�`�`�������������������
��3�:�.�#� ��
����l�b�l�r�x���������x�l�l�l�l�l�l�l�l�l�l�b�\�U�Y�d�{ŔŠ����������������ŭŔ�n�b�5�0�(�(�(�4�5�A�J�N�R�O�N�E�A�5�5�5�5�5������������&�$�������������������¦°ª¦�f�_�]�f�r�����������r�f�f�f�f�f�f�f�f�n�k�a�U�U�a�n�n�zÇÈÓàáìàÓÇ�z�nùìàÓÒÕßì����������������������ù�H�C�F�H�U�a�k�l�a�U�H�H�H�H�H�H�H�H�H�H��������������"�/�H�g�t�t�l�H�;������׾����������������ʾϾоʾþ����������������������������ûĻĻû����������������������������x�w�x�{�����������������������f�_�Z�M�A�?�A�T�Z�f�s�t�����������s�f�S�:�!����.�S�y���Ľ��(�L�S�J�*�����S����������������������������������������ŠşŔőŔşŠŭŹ��������������ŹŭŠŠ�Ϲ��������ùܹ�������'�'�������������������������������������������������߼�����������!�%� �������������۾޾����� �	����	�������=�8�=�=�I�V�b�n�o�o�o�b�V�I�=�=�=�=�=�=�e�`�Y�Y�`�e�m�r�~�����������������~�r�e�ɺź��������������ɺֺۺ����ֺκɺ��T�S�H�;�8�;�>�Q�T�a�m�z������|�z�m�a�T��������	���(�(��������������������A�=�7�8�<�A�M�S�Z�f�s�����|�s�f�Z�M�A������x��������������������������������Ŀ¿��������ĿѿԿݿݿݿؿѿĿĿĿĿĿĻ����ܻ׻лʻϻлܻ�������������������������Ŀѿݿ߿ݿٿѿĿ�������������ƎƈƁ�zƁƎƚƧƳƽ������ƳƧƚƎƎƎƎ�����_�M�E�D�M�\�l���ûܻ����	��ܻл��C�@�:�C�N�O�Q�\�h�m�h�_�\�O�C�C�C�C�C�Cā�}�u�h�[�O�B�@�C�W�[�h�tĎēĔĒēčā��������������������
���������a�Y�X�\�a�m�m�y�z�{�����������~�z�m�a�aE�E�E�E�E�E�E�E�E�E�E�E�FF E�E�E�E�E�E����������������������	�����	���������A�5�-�+�5�A�s�����������������s�g�Z�N�A�M�O�W�h�tąčěĦıĵĵĲĪĚā�t�h�[�M�����������������������������������������	�����$�'�+�&�$������������������	�����.�5�A�D�;�7�)��ĦěĜĤĦĳĿ����������ĿĳĦĦĦĦĦĦŠŝŔŇőŔŠũŭŹ��������������ŹŭŠ�0�(�#����#�&�0�<�I�U�b�m�y�e�U�I�<�0¿»²¤¦²¿����������������¿�z�x�n�m�n�n�p�zÇÓÕÙÓÓÇ�|�z�z�z�z�x�n�t�x�����������������x�x�x�x�x�x�x�x���������ûлܻ��ܻлû���������������������ݿڿݿ�����(�5�9�A�I�A�5�(���g�d�g�i�s�t�����������������������s�i�g��������'�4�@�F�H�M�O�N�H�@�4�'� N � " < = � E h = > ] W @ T & 8 I W e { T g < ` K 3 L 7 J Z g w 4 I G i L ! 4 B  &  ) : @ 4 9 z Q > C ` X 3 E ~ Z ~ K N /   3 G 7 J S  ^ S    �  z  ,  a  0  m  N  �  �  u  �  �  z  8  �  �  0    �  �  �  p  �  `    f  �       R    �  �  6  [  x  x  :  �  6  �  �  �    �  s  �  �  �  c  �    �  8  k    �  z  B  {  Q  �  �  �  |  �  |    -  H  &  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  D^  �  �  �  x  l  a  V  J  >  1  !    �  �  �    8  �  �  J  @  +          +  *  !    
  �  �  �  �  �  �  k  O  4    )  ;  G  N  Y  a  b  ^  T  @    �  �  �  Y  !  �  l  �  (  1  9  @  C  C  A  ;  .      �  �  �  �  e  /  �  �  )  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  x  r  l  f  _  Y  >  >  ?  @  A  C  K  S  \  d  g  f  e  d  c  U  B  0    
  
  	                       �       
            �  �  �  �  �  �  �  �  �  n  ]  K  3    �  �  k  5   �  x  |  �  �  �  �  �  �  �  �  �  �  �    %  7  G  W  g  w  �  �  �  �  �  {  Q    �  �  V    �  V  �  �  1  �     �  �    9  @  1    �  �  �  B    6      �  �  {  +  �   �  �  �    j  P  6    �  �  �  �  |  _  B  "  �  �  {   �   >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  {  ~  �  �  �  P  �    G  �  �  �  �  �  i  p  J  �  �    �        �  �  �       �  �  �  �  �  �  y  V  *  �  �  U  �  C  �   �  1  /  .  -  *  #      
  �  �  �  �  �  �  �  �  �  �  �  l  f  R  3    �  �  �  `    �  �  4  �  9  �  5  �  �  H  �  �  �  �  �  �  �  �  �  �  �      '  8  I  [  l  }  �  �  &  5  )    �  �  �  a  $  �  �  5  �  �  �    |  f  �      
      �  �  �  �  �  �  �  �  e  J  -     �   �   �  U  R  O  K  H  E  A  >  :  6  1  -  %        �  �  �  �  �  �  �  �  �  S  '  �  �  �  �  ~  n  f  ^  V  8    �  �  �  �  �  �  y  h  V  C  1      �  �  �  �  �  �  �  7  |  M  >  (  �  �  �  �  V  -    :  6       �  �  �  �  �  i  :  <    �  �  �  �  �  �  y  O    �  �  F  �  �    �  �  �  �  �  �  �  �  �  �  �  �  }  f  K    �  �  s  =    �  r  g  S  <    �  �  q  %  �  �  Y  +  �  �  �  r  {  �    +  =  G  =  3  (    
  �  �  �  �  |  R  &  �  �  �  .   �  R  Y  _  f  m  s  u  w  y  {  t  f  W  I  :  +      �  �  �  �  �  �  �  �  �  �  �  p  V  A  ?  1      �  �  m  s  <  )       �  �  �  �  �  }  i  S  ;       �  �  �  U    �  w  �  �  ^  $  �  �  �  D  �  %    $  �  k    �  �  Y  �  �  �  �  �  �  �  �  x  [  <    �  �  �  �  v  G     �  4  1  -  .  0  ,  &           	    �  �  �  H  �  �  c  �  �  �  �  �  �  n  ]  W  G  +  �  �  R  �  !  F  ?  �  �    �  �  �  �  �  �  r  ]  H  4  #       �   �   �   �   �   �  �  �  x  X  o  �  �  �  �  �  �  �  y  B    �  V  �     I  O  W  t  �  �  �  u  r  �  �  �  �  �  Z  (  �  o  �  J    ;  +    
  �  �  �  �  �  �  �    n  ^  M  >  .    �  �  �  �  �  �  �  g  I  +    �  �  �  �  �  l  J  *  -  7  D    /  @  G  B  4  "    �  �  �  �  d  %  �  s    �    t  �  �  �  �  �  �  �  �  �  �  �  �  w  j  [  H  &    �  =  �  �  �  �  �  �  �  �  �  �  �  �  x  m  _  Q  B  4  &    t  �  �  x  d  W  C  %  �  �  f    �  _  
  �    "  �  =  �  �  �  �  �  �  �  r  d  V  H  ;  ,        �  �  �  <  �    y  n  b  T  ?  )    �  �  �  �  m  O  4       �  Q  a  W  L  B  7  '      �  �  �  �  �  �  �  �  �  z  j  Z  =  /      �  �  �  �  �  g  E  !  �  �  �  8  �  �  N    �  �  }  s  i  _  S  G  :  .  
  �  �  Q     �   �   �      \  T  c  a  V  A  #    �  �  �  X  (  �  �  �    T    �             �  �  �  �  �  �  �  �  �  �  �  �  �  y  o  f  d  Z  N  B  8    �  �  �  �  �  �  a  /  �  �  �  ?  �  �  �  �  �  v  f  Y  W  U  S  P  I  <  0  #    
  �  �  �  �  !  &  '  "      �  �  �  �  `    �  �  d  ;    �  �  �  l  �  u  Y  6      	  �  �  �  w  -  �  ]  �    4  2    A  9  ,      �  �  �  v  D    �  �  U    �  {  /  �    �  �  �  �  }  W  +  �  �  u  -  �  �  \  !  �  �  o  %   �  :  �  3  F  <    �  �  e    �  W  
�  
  	D  h  �  q  �  �    u  k  a  W  N  D  5  $      �  �  �  �  �  �  �  t  b  �  �  �  �  �  �  �  �  }  i  U  A  )    �  �  �  �  �  m  �  �  �  �  �  �  �  �  �  �  z  f  P  <  .    �  �  �    �  �  q  `  J  5      �  �  �  �  �  �  �  �  �  h  K  .  \  f  i  _  R  ?  (    �  �  �  �  y  H    �  �  !  z   �  V  T  O  F  9  (    �  �  �  �  �  W  '  �  �  6  �  �  6  &  -  0  5  @  5  )  !      �  �  �  �  s  c  s  �  �  v  �  �  �  �  �    c  E  %    �  �  �  b  ,  �  r  �  B  �  G  7  &      �  �  �  �  �  l  O  1    �  �  �  �    ]  /      �  �  �  �  �  u  g  Z  M  ;  )    �  �  �  p  E  �  �  �  �  �  w  [  >    �  �  �  v  -  �  �  y  S  �  �  "  $  !    �  �  �  �  �  �  h  H    �  �  �  Q  �    Z  �  ~  h  K  .    �  �  �  �  i  ?    �  �  o    �    