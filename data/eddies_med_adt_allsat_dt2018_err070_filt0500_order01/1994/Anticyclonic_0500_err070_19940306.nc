CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��hr�!      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P0J      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =�hs      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @E\(��     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @v|�\)     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @R�           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @�/�          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >�      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�6�   max       B,fZ      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�d�   max       B,he      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�o   max       C���      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?   max       C��F      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          q      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          +      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�(�\)   max       ?�M����      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��t�   max       =�1      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @E�
=p��     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�\    max       @v|�\)     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @R�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @�@          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?   max         ?      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�X�e+�   max       ?�H��     �  T                  C      
   N   
   
   	                                       	   	               	             
   ,               
   (            3   )   q   !               1   O               OW_�N{��N(G�N��%N��O��N���N�i�P0JN��NN�{�N��EN�pN�ΚN$��M��O�%N�3�OS{�O�O�"N#
�O)Y]N	`
N�<�O+�MO��N�7BN)�fN�nO�AO���O�S�PE�N�5kOvu�O��On�N��N��N���O�>N:4N��fN���O�'KP$GNO��6O��O�NM�=O+5-O�6O���OێN�D�N�BN$�N�`N@�=���ͼo��`B��`B�D���D���o;��
;�`B<t�<t�<#�
<49X<49X<49X<49X<D��<T��<T��<u<u<u<�o<�C�<�C�<�t�<�t�<�t�<��
<��
<��
<��
<�1<�1<�1<�j<�j<���<���<�/<�<�<�<��<��=+=+=C�=�P=,1=,1=e`B=m�h=u=�%=�%=�o=�O�=�O�=�hsY[gtwrt���������ti\Yhist�������thhhhhhhh������������������������������������������������������������c`bht�������������pcMNQW[dgtzutigf[NMMMM?==>BBORY[\[VONB????	"/;HR[`aLH/"������������������������������������������������������������$)35;?<5)����������������������������������������r{�����������{rrrrrr�����
#/4<A?<2#
 ������������������������������������������������������������������6FSROB)�����WV[dhmmnh[WWWWWWWWWWKHOO[hnx|~����th[PK$������������������������
 )-.*)�����������������������,'&0<IOJIA<0,,,,,,,,��������������������������������������������
#/4433/&#
��?OUans~�����znaUK@<?�������
�������������"!
������������������������  #/<HOUX_^UHC</# �����������������������������
�������������������������������������������������������������������	!##��@;ABMO[\[OOB@@@@@@@@<BOS[ehtyttsjh[OGB<<{w����������������{{�������
��������)5BKNOPNB)�����������

������������� � ��������
#08600*'#
Ybfn{����{nbYYYYYYYY�����������������������������������z}����������������}{}����������������})-02/)(! !"###/02443/#!!!!"""-'"_ZX[anuz|{zwna______;8:<HTTMH<;;;;;;;;;;Çà÷÷ù������ùìàÓÇ�z�n�f�]�n�zÇ�L�Y�e�i�o�e�e�Y�W�L�G�C�L�L�L�L�L�L�L�L��'�3�4�5�3�'������������������������������������������������������ĿѿؿԿѿʿĿ������������������ĿĿĿ��B�O�[�h�t�y�z�q�h�[�B�8�6�)�'����6�B�m�q�y�����������y�m�j�`�_�`�h�b�m�m�m�m�l�y���������������������y�m�l�i�l�l�l�l�"�;�T�a�p�x�����v�a�T�H�;�������������"��(�0�5�<�9�5�*�(���
���������ѿݿ�����������ݿѿĿ������ÿĿ̿ѿѻ�����������������޻����׾����	��	��������׾;̾վ׾׾׾׿�"�.�.�6�8�.�+�"��������������ûлѻлλŻû����������������������������������������������������������������`�m�y�����������~�y�k�`�T�L�G�C�E�I�T�`����*�6�8�;�6�5�*�������������ܹ�������
����ܹȹù��������ùϹ��"�/�;�H�I�R�R�H�G�;�/�"�������"�"���	���/�9�9�4�"�	���������������������-�:�@�F�L�F�:�-�"�#�-�-�-�-�-�-�-�-�-�-�ʾ׾�����׾ʾ��������������������ʾZ�Z�f�h�f�a�Z�M�K�M�Q�Z�Z�Z�Z�Z�Z�Z�Z�Z����!�%�!�������������������h�u�zƁƎƖƚƟƚƓƁ�u�h�g�a�Y�R�\�a�h������������	�	�� �������������������Ҽr�������������}�r�n�l�r�r�r�r�r�r�r�r�4�A�L�J�A�A�@�4�*�-�4�4�4�4�4�4�4�4�4�4�f�s�u�~�s�g�f�f�`�[�f�f�f�f�f�f�f�f�f�f���(�2�3�3�7�8�5�(��������
���N�g�t�x�q�g�N�A�5�(�"�����"�(�5�@�N�	�"�/�6�G�Q�O�H�;�4�/�"��	�����������	�s����������
�
�㾱�����s�Z�M�F�I�Y�`�sE�FFF$F+F1F=F=F=F9F1F+F$FFE�E�E�E�E�������������������������������޽��нݽ�������ݽн������������������T�`�m�r�{�����v�m�d�`�T�G�E�?�?�>�@�G�T���ʼּ׼�ݼڼּʼ���������������������������������������������������ûǻлջػлû������������������!�-�:�l���������x�a�F�:�!��������	�!��������������������������H�J�P�U�\�a�e�i�a�X�U�H�<�;�8�9�<�A�H�H�<�H�N�U�X�U�R�H�B�<�/�#�#�!�#�$�/�3�<�<āčĚĳ������������ĳĚčā�m�h�^�g�~ā�#�<�U�b�x�x�r�o�d�U�I�0�#�
���������
�#D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DvDlDoD{D��y���������������������y�l�g�c�Z�`�l�u�y������!�-�+�(�!�������������������������������������������������������������������������������������������������������������������¿¿¼¿����Óà������������������ùìàÓÐÈÈÐÓ���������ʼ��� �������ּʼ������������e�r�~�������������~�r�f�e�Z�e�e�e�e�e�e�I�V�b�o�{ǈǈǈ�|�{�o�b�V�I�?�@�I�I�I�I�
���#�#�#��
�� �
�
�
�
�
�
�
�
�
�
�����ûлػջлǻû���������������������EEEE E(EEEEEEEEEEEEEEE H D 8 E n O @ F < $ ; ? C F Y � 7 O [  T B U X + : ' 8 : C A 2  q k   N 0 $ P L q M i < :   2 . v 3 H - U K  D ' -  �  �  -    �  �  �  �  .    *  %      <  �      
  :  �  =  �  <  �  �  7  �  T  1  _  D  �  "  �  �    �     9  '  �  :    �    �    %  5  �  |  2  =  l  �  �  6  �  Y�ě�;D���o;�`B;o=�%;�o<u=�1<���<�t�<�t�<�t�<u<�o<T��=��<��
<�<�1=t�<���<�h<��
<���<���=\)<���<�9X<�9X<�`B=49X=#�
=Y�<�=�+=0 �=T��=<j<�=�w=�\)=t�=��=D��=�1=��>�=�\)=Y�=ix�=�\)=�hs=�/>bN=�\)=�O�=��=���=�9XB	�B~aB!�^B!�"B ��B
��B	�B�A�^B&B�WB!�B�4B! �B!��B(�IB\3B�LB$�B�;B�\B�.B�-B�IB�VB̧B3B&�B��B�BQ*Bm�B�FB'B.�B��B ��B��B"rMB��B6WB��B��B;yB��B"�Bq�B��B,fZB$�nB(MdB�wB��B�B��B��B�A�6�B��B%B	�3B�B!_�B".LB �rB:B	;�B��A�ocB>JB�6B!)�BȿB �cB!fpB)�+B7�B|�B@B�dB�MBX�B��B�iB ?KB�ABЎB&,BIB��BB?�B��B@>BJ�B�0B �iB7�B"AcB��B��B��BP
B+�B��B��By�B��B,heB$�^B(@�B��B�B@ B��B<�B+�A�d�B��B?�A��S?� �?�V@pAx�A��eAl�A/A���A��A}O�@�t�AVt�A^+�@��&@��Aj<�A�\�?�oA��^A��l@yw�APv�A>�]@_��B�A��@�a�A:OeAA��A���A���A�AAP*�C���A�>A'ԓAh-@���A��@��@z��@��HA�T�A�M�A�>�A��C���A��A	�@S�>A�חA���A�X@���@�-B�A�"b@�>�C�jfAʂl?�y�?�@_mAx|�A؎�Am�A�A�w/A�� A~�9@��AT�.A^͆@� @@풒Aj)(A�`�?A�5�A��V@z]�AP��A?@c� B��AЪ�@�mA:̤AB��A�E�A�g A��sAQC��FA�NA(��Ah��@��A���@��@tw@��MA��CA�?�A���A�ZC���AtA	/@T\A��A�� A�u A =9@��B�A�^@�e C�h�                  C         N      
   	                                       
   	               	         !   
   ,               
   )            4   )   q   "               1   O                                 %         +                                    %                                       +         %                           #   '                        %                                                                                                                              %                              '                                       NܪNW�N(G�N���N��OxBxN]�N�i�O��N�C"N�]�N��EN�pN�ΚN$��M��O�%N�3�Nѓ)O�O��N#
�O�7N	`
N�<�O+�MN���N�7BN)�fN�nN�%Os*}O'|%O3G2N�5kN�u9O��Oa��N��N��N���OO
lN:4N��5N��O���P��O��O��N��>NM�=N��>O�6O0��O�fYN�D�N4�N$�N�`N@�=  �  �  �  �  �  	c  W  5  	;  �  <  !  �  :  �  \  /  �  C      �  �  �  \  �  *  u    c  �  6  V    �  q    �  �  	  �    h  �  �  f  S    �  P  D  �  w  :  
�  ]    �  �  ڼ�t���`B��`B��o�D��<�t���o;��
=��<#�
<49X<#�
<49X<49X<49X<49X<D��<T��<���<u<�1<u<�C�<�C�<�C�<�t�<ě�<�t�<��
<��
<�1<�j<���=+<�1=,1<�j<�/<��<�/<�=�P<�=o=C�=H�9=t�=�1=H�9=<j=,1=u=m�h=��=��
=�%=�+=�O�=�O�=�hsd`adgt��������tpgddjjtt�������tjjjjjjjj������������������������������������������������������������ojmtv�������������}oQSZ[\gtxtrg[QQQQQQQQ?==>BBORY[\[VONB????"/;HNNKH?;/"������������������������������������������������������������$)35;?<5)����������������������������������������r{�����������{rrrrrr�����
#/4<A?<2#
 �����������������������������������������������������������������	*6;BDJGB=)�WV[dhmmnh[WWWWWWWWWWLJOR[hmtv{|tnh[[ROL$������������������������
 )-.*)�����������������������,'&0<IOJIA<0,,,,,,,,�������������������������������������������
#,/111/$#"
���H@CHRUanq|���znaURH������� 


���������������

�����������������������.(&*/<HLRNHD</......�����������������������������	���������������������������������������������������������������������@;ABMO[\[OOB@@@@@@@@ABOT[fhtutrohb[OLBAA����������������������������������������5BGKOLB)#��������� 

��������������������������
#%020%# 
Ybfn{����{nbYYYYYYYY�������������������������������������������������������������������������)-02/)(#"!#/221/###########"""-'"_ZX[anuz|{zwna______;8:<HTTMH<;;;;;;;;;;�zÇÓàæëêàÓÈÇÆ�z�o�n�i�n�w�z�z�L�Y�e�g�l�e�[�Y�Y�L�H�E�L�L�L�L�L�L�L�L��'�3�4�5�3�'������������������������������������������������������ĿѿؿԿѿʿĿ������������������ĿĿĿ��B�O�[�f�h�n�o�k�h�e�[�O�6�)�'�$�%�-�6�B�m�y�����������y�n�m�c�j�m�m�m�m�m�m�m�m�l�y���������������������y�m�l�i�l�l�l�l�"�/�;�H�T�[�e�e�[�H�;�/�"��	������"��(�,�5�:�7�5�(�'�������
�����ѿݿ������ݿѿ̿ʿѿѿѿѿѿѿѿѿѻ�����������������޻����׾����	��	��������׾;̾վ׾׾׾׿�"�.�.�6�8�.�+�"��������������ûлѻлλŻû����������������������������������������������������������������`�m�y�����������~�y�k�`�T�L�G�C�E�I�T�`����*�6�8�;�6�5�*�������������ܹ������������ݹܹϹƹϹѹܹܹܹ��"�/�;�H�I�R�R�H�G�;�/�"�������"�"���	��"�/�3�2�/�(�"���	���������������-�:�@�F�L�F�:�-�"�#�-�-�-�-�-�-�-�-�-�-�ʾ׾߾����پ׾ʾ����������������ľʾZ�Z�f�h�f�a�Z�M�K�M�Q�Z�Z�Z�Z�Z�Z�Z�Z�Z����!�%�!�������������������h�u�zƁƎƖƚƟƚƓƁ�u�h�g�a�Y�R�\�a�h���������������������������������������Ҽr�������������}�r�n�l�r�r�r�r�r�r�r�r�4�A�L�J�A�A�@�4�*�-�4�4�4�4�4�4�4�4�4�4�f�s�u�~�s�g�f�f�`�[�f�f�f�f�f�f�f�f�f�f��#�(�1�1�5�5�3�(�������	�����5�A�N�V�g�m�s�m�g�N�A�5�(�����'�(�5�	��"�/�<�H�C�;�/�$�"��	�	����������	�������ʾ׾��������ܾ׾ʾ�����������E�FFF$F+F1F=F=F=F9F1F+F$FFE�E�E�E�E�����������������������������������޽��нݽ�������ݽн������������������T�`�m�q�z����u�m�c�`�T�G�F�@�?�A�G�I�T���ʼͼּۼּּʼƼ���������������������������������������������������ûǻлջػлû������������������!�-�:�S�_�m�v�l�S�P�F�:�!�������!��������������������������H�I�O�U�[�a�c�b�a�`�U�H�>�<�:�;�<�G�H�H�/�<�H�R�N�H�?�<�/�&�$�'�/�/�/�/�/�/�/�/āčĚĦĳĴĿ��ĿĳĦĚčā�t�r�k�n�sā�#�<�I�U�b�p�r�g�]�U�I�#�
�����������#D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}D�D�D�D�D��������������������������y�w�o�u�y������������!�#�"�!�����������������������������������������������������������������������������������������������������������������������¿¿¼¿����ìù����������������ùìàÛÓÐÑÓàì�������ʼּ���������ּʼ��������������e�r�~�������������~�r�f�e�Z�e�e�e�e�e�e�I�V�b�m�n�b�V�I�E�I�I�I�I�I�I�I�I�I�I�I�
���#�#�#��
�� �
�
�
�
�
�
�
�
�
�
�����ûлػջлǻû���������������������EEEE E(EEEEEEEEEEEEEEE C C 8 A n = E F 6 #  ? C F Y � 7 O ;  . B 8 X + :   8 : C ? 2 " h k  N . # P L g M d :    > H v $ H 1 O K ` D ' -  �  �  -  �  �  
  �  �  >  �  �  %      <  �      �  :    =  2  <  �  �  �  �  T  1  .  �  b  �  �  �    �  �  9  '  �  :  �  �  �  m  2  9  �  �  �  2  �  q  �  >  6  �  Y  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  d  �  �  �  �  �  �  �  �  �  �  �  `  #  �  u  1  �  D   �  �  �  �    
      �  �  �  �  �  �  �  s  U  6    �  �  �  �  �  �  �  �  �  �  �  �  �  z  [  9    �  �  �  d  3  �  �  �  �  �  �  �  �  �  �  �  �  l  K  (    �  �  �    �  �  �      �  �  �  �  �  �  �  �  �  �  �  k  Q  6    <  �  �  	7  	P  	^  	c  	`  	R  	2  �  �  s     �  M  �  �  X  �  I  N  R  W  T  O  K  A  3  %    �  �  �  �  �  _  ?     �  5  -  &         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  0  �  �  	  	.  	9  	:  	.  	  �  �  L  �  P  �  �  c  �  �  �  �  �  �  �  �  �  �  �  g  G  '    �  �  �  {  G    �    *  4  ;  :  7  0  &      
  �  �  �  �  �  v  T  2  !      
      �  �  �  �  �  �  �  �  �  �  w  z  �  �  �  �  �  �  �  �  �  y  h  V  D  2  "    �  �  �  �  �  s  :  8  6  4  1  /  *  &  "           �   �   �   �   �   �   r  �  �  �  �  �  �  �  �  p  ^  L  :  '      �  �  �  �  �  \  b  g  m  s  x  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  /  +  %        �  �  �  �  c  :    �  �  I  �  �  O    �  �  �  �  �  �  �  �  �  �  }  g  M  3      �  �  �  �  #  7  6  /  %  $  1  B  ?  6  #    �  �  �  o  8  �  �  �    �  �  �  �  �  �  �  �  �  �  �  t  a  G  ,     �   �   �  �  �  �  �            �  �  �  �  �  U  %  �  �  o  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  R  �  �  �  �  �  �  �  �  �  �  x  h  V  C  ,    �  �  �  {  �  �  �  �  �  �  �  �  �  �  �  p  _  L  6    	   �   �   �  \  R  I  >  4  %      �  �  �  �  q  `  t  z  R  *  �  �  �  �  �  �  �  �  �  o  Z  C  -    �  �  �  �  �  �  {  [  �        !  '  *  )  #      �  �  �  �  �  q  j  �  �  u  u  t  t  t  t  r  p  n  d  Y  M  =  ,      �  �  �  �        
         �   �   �   �   �   �         
        c  `  ]  Z  W  U  R  O  L  I  F  B  ?  <  8  5  2  .  +  (  �  �  �  �  �  �    s  e  W  I  :  *    �  �  �  �  �  d  &  1  5  3  -  !    �  �  �  �  �  d  6  �  �    ?  �   �  �    .  E  T  V  M  :  #    �  �  �  �  �  �  M    �  M  �  �  �  �  �  �  �     �  �  �  �  a    �  Z    �  =  �  �  q  a  K  4    �  �  �  �  `  4    �  �  s  H    �  �  Q  �  �    3  P  c  m  o  p  d  F    �  z    _  �    �    
  �  �  �  �  �  �  �  v  k  a  U  =    �  �  �  a    �  �  �  �  �  s  ]  >    �  �  �  c  +  �  �    �    �  �  �  �  �  �  �  �  �  �  g  +  �  �  �  ^     �  �  O    	    �  �  �  �  �  �  �  �  �  �  �  �  �  �    v  m  d  �  �  �  �  �  �  �  �  �  w  e  O  7    �  �  �  9   �   �  �  �            �  �  �  �  X     �  �  <  }  �  �   �  h  a  [  T  C  1      �  �  �  �  v  Q  *    �  �  �  ]  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  y  w  t  m  �  �  �  �  �  �  �  �  �  �  �  �  c  ,  �  �  m  -  �  �  �  �  1  O  ^  f  b  X  I  3    �  �  U  �  �  3  �  z  N  R  R  R  P  K  F  B  ;  .    �  �  �  H  �  �  =  |  :  a  �  i  �  A  �  �  �      �  �  A  �  �    -  (  	�  d  �  *  [  ~  �  �  �  �  �  �  �  f  :    �  �  �  U  *  %        $  /  ;  F  P  K  @  *    �  �  �  �  �  �  �  x  D  4  !    �  �  �  �  �    Y    �  u  7  �  �  p  )  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  >    �  �  j  �  w  e  R  =  &    �  �  �  �  x  T  .    �  }  4  �  n   �      %  0  6  :  5  !    �  �  s  2  �  D  �  �  �  �  �  
}  
�  
�  
�  
�  
�  
�  
�  
�  
q  
l  
<  	�  	�  	&  �  �  �  �  �  ]  C  (    �  �  �  �  h  F  #          &  A  _    �  �  �  �    	  	  
    	    �  �  �  �  �  �  �  `  0     �  �  y  h  X  G  5  $       �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  Y  <    �  �  �  �  �  `  @  !    "  %    �  �  �  o  J  #  �  �  �  X    �  �  ^    �  �    �  @