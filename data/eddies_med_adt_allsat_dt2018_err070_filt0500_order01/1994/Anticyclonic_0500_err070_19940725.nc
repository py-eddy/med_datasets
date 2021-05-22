CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����l�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�   max       P��n      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �D��   max       =��      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @E��Q�     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��p��
>    max       @vj�Q�     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @P�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >49X      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�j�   max       B+��      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A� H   max       B,;I      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =��~   max       C��      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =G�   max       C��      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�   max       P#H;      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�D��*1   max       ?�v_ح��      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �,1   max       =���      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E��Q�     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(��    max       @vi��R     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�L�          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��-V   max       ?�t�j~��     �  Pl                                       "            (               d            j            �      +                           	   n      2         O   h            :         (   O�^�N�mN$Nu�Na�~OU�N�RN�ixN��O�D�N�7O���O�bN��O���O�WRPx�DN0��O�r5N@��O�P��O��Od�|O�mPw�mNm:O���NڇmP��nN�"\PX�O�@<NDުO'��M���M�N�
O֐Oͮ�N�dO�2N	�kOF�}O��N`S�O�T�O��NsD&Ow*N��O�}eN�18N��OyN�z�D���T����`B���
��o�o��o%   %   %   :�o;D��;��
;�`B<o<D��<D��<D��<T��<�t�<��
<�1<�j<�j<ě�<�/<�`B<�`B<�`B<�`B<�<�=+=+=+=C�=0 �=49X=<j=@�=D��=P�`=P�`=T��=Y�=]/=m�h=q��=��=��=�+=�\)=�\)=��=�j=����������������������HIRUbnprrpnnkbZULIHH������������������������

���������������������������jedhmz�����������zrj--/0<<<HUV`_USH<5/--����������������������������� ����������&0<IU\_[PID<0#		"/;@<;/)"	)5ONYWNB?5)`cdabcht��������th`/,/6;BOXTOKCB6//////RV`gt����������tka[R������������ �����	5F[�������t[IPG)�����������������������)5BHPUSN5 ��xqvz�����zxxxxxxxxxxpt���������������tp_`n��������������zd_�����
#/<GKHF>#
���KUY`nz��������zng\UK<888;>BNOV[]][YQNJB<�z�}����������������)57544)'orrt��������������to��������������������(LbQ[t�������xOB)
#/<<?<1/,#������'/0)��������������������(&/<HOJH</((((((((((A=@HUanzz��|znaULHAA��������������������soty�����tssssssssss����

���������������
 
�����5:96:?@8)	 ������������������������������

 ������������������������mnz�����������}ztqnm#+/<HMSQHD</(#!\chmt����|th\\\\\\\\������
$+022/)#
��������thecgt��������&)/6>BOPONB60)&&&&&&��������������������������������������������������������")66:=96+)&��������������������30/26;HTX^`a^TJH@<;3!#./<=DHGD</.(##!!�����������������������ĿĳĪĳĹ���̻��������������������x�x�x�y�����������������������������������������������������F�S�_�f�h�_�Y�S�F�C�:�7�:�D�F�F�F�F�F�F�ûƻлۻлλû����������ûûûûûûû��;�H�T�a�e�j�m�p�u�m�a�T�H�;�/�+�(�#�/�;�������������������������������������������������������������������������������������������f�r�����������������������e�Y�Q�M�U�f�H�T�a�a�j�m�s�y�v�m�a�`�Y�T�N�H�C�C�H�H��"�;�G�J�U�\�`�T�G�.�%��������	���M�Z�f�s���������������Z�M�4�,�4�;�?�M�����������������������������������������H�T�a�e�g�c�[�H�:�8�/�"�.�$���"�/�<�H������"�2�:�>�;�5�/�"��	� ���������������������������������g�S�L�J�@�A�s�������s���������������������������s�m�s�s�s�s������$�4�?�<�0��������������������������������������������������������ٿ.�3�=�G�K�H�G�A�;�.�"���	���!�"�*�.�ù������1�<�7�'����ܹù��x�T�X�z������5�A�I�M�M�=�9�/�(��������������(�6�A�N�Z�_�Y�N�A�5�(��������$�(�h�tāčĚĦĲĮĦĚčā�t�k�h�[�X�[�`�h�Y�������ļ���������f�@�'����� ��'�M�Y�M�Z�e�c�]�Z�Q�M�A�=�A�K�M�M�M�M�M�M�M�M�������������ĿѿԿԿĿ��������������{���`�m�y�����������y�m�c�`�T�O�T�V�T�\�`�`�ܼ�4�Y�����ʼ������Y�M�4���ܻ��������T�a�m�m�j�a�a�T�O�H�E�H�J�M�T�T�T�T�T�T�	�"�/�6�@�T�`�[�Q�;�"���������������	�5�A�D�/�+�,������׿̿ʿ˿ѿ����(�5���������������������������������������"�/�3�5�9�7�/�-�"��	�����	����g�s�������s�q�g�Z�U�Z�d�g�g�g�g�g�g�g�g�)�5�>�B�K�B�5�)�(�&�)�)�)�)�)�)�)�)�)�)�/�1�3�1�/�(�$�#�������#�,�/�/�/�/�ʾ׾�����	���"�&�"��	����Ӿɾ��#�0�<�K�O�N�P�I�0�#�
���������������
�#����(�.�4�A�E�B�A�4�,�(����
���D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DxDyD�D�D�����������������������������������������������ٺֺɺ��������������������ɺ�A�B�M�Q�Z�\�]�]�Z�M�A�?�4�-�)�(�'�(�4�A�-�:�@�F�L�F�F�:�4�-�%�$�-�-�-�-�-�-�-�-EiEuE�E�E�E�E�E�E�E�E�E�E�EtEiEdEbEcEfEi�����������������)�8�A�C�C�?�6�)�����������
�����������������l�q�y���������������y�l�`�]�T�U�`�g�d�l�����������
��������������������������#�/�<�H�L�K�G�@�/�#��
�����������
��#�����������������������~�x�r�q�r�w�~����ÇÓÕÙÝÓÇ�z�s�p�z�|ÇÇÇÇÇÇÇÇĦĿ����������
����������ĳĮĦĚēĦ�{ǈǔǘǔǔǉǈ�{�o�b�V�O�T�V�b�o�q�{�{ ) 3 T 2 ? % $ 0 N @ m > P S V $ k u J # 0 L H # T B B ] Q g Q : X T B � N g c M W ; ; 9 H R ) # I 3 < ( " 4 g S    �  ?  s  t  �    �  0  =  0  Q  X  �  Y  .  �  �    M  G  �  �  �  m  ?  v  d    �  �  �    �  �  J  $  �  �  �  �  y    �  <  �  �  �  �  L  �  �    �  u  ������`B�D����o;�`B<#�
<ě�;�o<49X<�j<t�<ě�=��<T��<���<���=T��<�C�=C�<�1=o=���=@�='�=H�9>1'=\)=�w=�w>49X=C�=��=u=�P=0 �=t�=@�=T��=�%=��P=ix�>$�/=]/=��`=�C�=y�#>C�>%�T=�\)=�j=��
>o=�{=���>$�=��B@0B'�B"��B$8�B �B :B�9B�JB)�B&<:A�j�BPB��B'�B
GB��B	��B֭B�xB �B�B�jBM�B�B
�BB��B:{B��B3B��B�vB$B��B1)B ��B
�B0�B�kB@�BbB�-B#SB�Bb�BoB�
B
��B�B+��B ŏBvB4�B!ȑA�bKBd~B@1B'�dB"��B$A>B ȁB �B��BڷB@�B&<HA� HB�B�B9}B
B��B
�WB�uB�B CB�pB�BB~#BC�BA}B)�B��B6BAYB��B��BѰB�`B�FBAB ��B	�fB*�B�lB �B?5B��B#G�B�XB@�B>:B��B
_�B��B,;IB5qB?�B?�B":�A�r�BR�A���@��Z@���@�qJ@�t�A��	A�,#A��
A�!{@�5A�]wAaACk�A"A�D<A���A�$�A�z~Bw�BM�A`Ih=��~A�X}A�mA�^@�ީA=�At�iAj�^@���A��A��MA�@vA��A�*aA�9A���A���AX�A�Y�A6�>C��AJa'@*�YA;RO@y�7C��A�#@YT�A4�AѼ�A�@��AɎA�BHB�A��@��P@�W�@��@�_A�r@A�u�A���A�|-@���A�RAa|�AD��A"��A�lA���A�iNA�~OB�YBz�A_Y=G�A�}�A��A���@�K�A?!�As�Ak @��MA�q�A�e!A���A��cA��A�{A�^A�wtAVڄA���A6��C���AJ��@+��A9�@|<C��AՀ@Tf�A��AѺ�A���@��AɁ�A��B�c                                 	      "            )               e            k            �      ,                  	         
   o      3         O   h            ;         (                                                      ?      '         =            5            I      '   '                     !                        !            #                                                               )               '            '                     '                                                                     O!M�N�mN$Nu�N=V�O+��N{v6N�ixN�)OQ\�N��1O�0:O h	N��O�3O���P#H;N0��O���N@��N��O�a�O�	lON�cN��P��Nm:O���N��O��N�"\O�kO�@<NDުO'��M���M�N�
O֐O���N�dO��N	�kO&LmN�5�N`S�O�wIOo˞NsD&Ow*N��O��YN�18N��O2�DN�z  �  !    A  �  O  %  �  '  W  3  A  =  �  �  �  �  �  �  d  8  

  A  ~    [  =      I    D  �  �  q    V  �  �  �  s  �    
3  2  �  B  q  �  �  �  	�  u    	0  ��,1�T����`B���
�D��%   ;�`B%   :�o;�o;D��;�o<e`B;�`B<T��<T��<���<D��<�t�<�t�<�9X=y�#<���<ě�<�h=q��<�`B<�`B<�=���<�=@�=+=+=+=C�=0 �=49X=<j=P�`=D��=ě�=P�`=ix�=]/=]/=�t�=� �=��=��=�+=��w=�\)=��=���=����������������������HIRUbnprrpnnkbZULIHH������������������������

���������������������������lhglmz�����������zul226<HITRH<2222222222����������������������������������������"!#-05<LUXVPI?<0&"		"/;<;7/%"					)5BMWWVNB>5)firt����������tomkhf/,/6;BOXTOKCB6//////c\^fgtx���������tkgc��������������������>6@N[g��������[TVTN>��������������������")35BLNMOKB5xqvz�����zxxxxxxxxxx��������������������mmqy�������������zpm����
#/<@FD<#
����_UV[bnz��������znja_?;:;@BNQW[[[[VNB????��������������������)57544)'orrt��������������to��������������������4136;BO[hty}{pbWOB94
#/<<?<1/,#�����!$#�������������������(&/<HOJH</((((((((((A=@HUanzz��|znaULHAA��������������������soty�����tssssssssss����

���������������
 
�����)1426;;85)����������������������������

	�������������������������wsqqqz������������zw!#-/<HKPHHC</#\chmt����|th\\\\\\\\������
&+-.+#
���pkiipt������������tp&)/6>BOPONB60)&&&&&&����������������������������������������������
����������")66:=96+)&��������������������643359;CHT[]^ZVTMH;6!#./<=DHGD</.(##!!������������������������������ļĿ�����ػ��������������������x�x�x�y�����������������������������������������������������F�S�_�f�h�_�Y�S�F�C�:�7�:�D�F�F�F�F�F�F�ûûлػлͻû��������ûûûûûûûû��;�H�T�a�b�g�m�m�q�m�a�T�H�;�/�.�+�'�/�;������
�����������������������������������������������������������������������������������������f�r������������������������o�Y�W�Q�Y�f�T�U�a�h�m�p�u�n�m�k�a�\�T�Q�H�S�T�T�T�T��"�;�G�T�[�\�T�G�;�.�&���	����	�������������������s�f�`�Z�J�J�M�Z�f�s������������������������������������������;�H�T�a�a�b�a�\�T�S�H�D�;�/�-�&�&�/�4�;������"�0�9�<�;�2�/�"��	�����������������������������������g�[�S�S�P�Z�g�������s���������������������������s�m�s�s�s�s�������$�,�0�2�1���������������������������������������������������������ٿ�"�.�;�G�A�;�4�.�"�������������ùϹܹ���������ܹù����z�v��������5�A�E�K�J�;�6�*�(�������� ��
���5��(�5�A�N�Y�]�Z�W�N�A�5�(��������h�tāčĚĞĞĚčāā�t�r�h�]�d�h�h�h�h�Y�f��������������r�@�4�)����&�4�M�Y�M�Z�e�c�]�Z�Q�M�A�=�A�K�M�M�M�M�M�M�M�M�������������ĿѿԿԿĿ��������������{���y�~�������}�y�m�`�^�T�P�T�X�Y�`�m�r�y�y�����'�1�<�C�G�E�@�4�'���ܻջѻܻ��T�a�m�m�j�a�a�T�O�H�E�H�J�M�T�T�T�T�T�T�"�/�;�H�O�S�L�H�;�/�"��	� � ��	���"�5�A�D�/�+�,������׿̿ʿ˿ѿ����(�5���������������������������������������"�/�3�5�9�7�/�-�"��	�����	����g�s�������s�q�g�Z�U�Z�d�g�g�g�g�g�g�g�g�)�5�>�B�K�B�5�)�(�&�)�)�)�)�)�)�)�)�)�)�/�1�3�1�/�(�$�#�������#�,�/�/�/�/�ʾ׾�����	���"�&�"��	����Ӿɾ��
�#�0�<�E�K�H�<�0�#�
�����������������
����(�.�4�A�E�B�A�4�,�(����
���D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������������������������������������������������ɺֺ�޺׺ֺɺź������������������4�A�M�O�Z�[�\�\�Z�M�B�A�4�.�)�(�4�4�4�4�-�:�@�F�L�F�F�:�4�-�%�$�-�-�-�-�-�-�-�-E�E�E�E�E�E�E�E�E�E�E�E�E�EuEiEeEgEmEuE�����)�6�9�=�=�8�6�)���������������������
�����������������l�q�y���������������y�l�`�]�T�U�`�g�d�l�����������
��������������������������#�<�H�G�C�=�/�#��
�������������
���#�����������������������~�x�r�q�r�w�~����ÇÓÕÙÝÓÇ�z�s�p�z�|ÇÇÇÇÇÇÇÇĿ������������
��������������ĿĶĳĿ�{ǈǔǘǔǔǉǈ�{�o�b�V�O�T�V�b�o�q�{�{  3 T 2 ? ( # 0 K B d : D S 4   H u . # . 4 F  E $ B ] \ G Q  X T B � N g c O W + ; ( G R '  I 3 < * " 4 P S  X  �  ?  s  X  x  ~  �  	  �  �  7  b  �  W       �    M  �    a  �  �  �  v  d  �  �  �  �    �  �  J  $  �  �  w  �      ^     �    �  �  L  �  .    �  �    ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �  �  �  �  �  �  �  �  �  �  �  }  X  )  �  �  u  1  �  �  !                  �  �  �  �  �  �  �  �  w  [  >      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  A  A  @  ?  ?  <  5  .  '       �  �  �  �  �  �  �  t  `  �  �  �     	        �  �  �  �    G  
  �  w  *  �  �  =  D  J  N  O  N  I  @  4  !  	  �  �  �  �  e  >    �  �  5  n  �  �      $  !        �  �  �  �  �  �  r  \  l  �  �  �  �  �  �  �  �  �  �  �  �  �  �    u  i  ]  Q  E    "  '  &      �  �  �  �  �  �  �  o  J  
  �  �  K    4  E  P  V  S  6      ?  ?    �  �  �  �  �  �  �  y  f         '  .  1  )         �  �  �  �  �  �  s  \  J  8  (  A  :  2  (         �  �  �  �  �  c  7  �  �  Q  �  �  �  �    /  9  <  6  "    �  �  �  d  9       �    1  �  �  �  �  �  �  �  �  �    k  T  8       �  �  �  �  y  b  )  a  �  �  �  �  �  �  �  �  �  h  L  -  	  �  �  �  K    �  �  �  �  �  �  �  �  �  z  c  G  '    �  �  �  k  >   �  ?  9  L  y  t  W  ?  *  2  '    �  �  �  s    �  P  �  r  �  �  �  �  �  �  �  �  �  �  �  y  f  Q  =  (     �   �   �  r  �  �  �  �  �  �  �  �  �  l  K  (    �  �  �  m  �    d  X  L  @  4  (      	     �  �  �  �  �  �  �  �  �  �  &  (  -  7  7  6  4  -  #      �  �  �  �  �  x  N  "   �  	�  	�  	�  	�  	{  	�  	�  
  
  	�  	�  	v  	  m  �  �    �  �  �  :  @  ?  8  /  &        �  �  �  �  �  |  Q    �  [  �  g  }  ~  }  ~  ~  z  s  i  [  F  (    �  �  �  u  R  B  :  �  �         �  �  �  �  �  a  9    �  �  g  %  �  �  >  	�  
�  
�    ?  T  X  <    
�  
`  	�  	�  	  g  �  �  ?  �    =  7  1  (      �  �  �  �  �  �  �  l  W  B  ,    �  �    �  �  �  �  �  �  {  ^  B  *    �  �  �  {  e  k  y  �  �              �  �  �  �  �  �  �  d  0  �  �  �  R  N  �  	�  
�  �    �  �  &  E  G  6  �  x  �    	�  �  ~  %        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        -  9  A  @  2    �  �  {  &  �  q  	  �  �  �  �  �  �  �  �  v  X  7    �  �  g  %  �  �  0  �  %   �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  w  k  \  L  =  .  q  o  n  m  l  g  a  [  S  K  A  4  &    	  �  �  �  �  �      	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  V  U  T  S  R  P  N  L  J  G  E  B  @  =  ;  8  7  5  3  1  �  �  �  �  �  �  �  �  �  �  �  p  W  ;    �  �  �  @  �  �  �  �  `  8    �  �  �  �  �  t  D    �  �  q  5  �  �  ~  �  �  �  �  �  �  �  w  i  Z  G  3     �  �  �  P  �  �  s  i  `  i  t  [  7    �  �  �  Y  &  �  �  �  N     �   �  h  �  ]  �    _  �  �  �  �  �  N  �  A  {  �  v  M  	X  �               �  �  �  �  �  �  �  �  �  s  F     �   �  	�  
   
2  
+  
  	�  	�  	�  	�  	Q  	  �  T  �  O  �          -  1  *            �  �  �  �  �  �  �  m  U  <  6  g  �  �  �  �  v  a  K  4      �  �  �  �  �  �  w  a  F  ,  �  �  )  A  =  )    �  �  m    �  I  
�  	�  	7  c  �  �  �  �  �  -  Y  m  q  k  Y  1  �  �  /  �  �    
�  	�  �  �  �  �  �  �  �  �  �  �  |  m  _  Q  B  4  '        �  �  �  �  �  �  �  �  �  m  T  2    �  �  T    �  �  C  �  �    �  �  �  �  �  �  }  i  U  ?  )    �  �  �  �  �  .  m  �  �  	B  	u  	�  	k  	A  	  �  	  	  �  v    �  0  �  i  �  �  �  u  j  ^  L  3      �  �  �  �  X  *  �  �  �  S    �  p    �  �  �  �  �  �  �  �  q  X  .  �  �  �  L    �  l    �  	  	$  	.  	#  	  �  �  �  �  n  7  �  }  �  E  e  r  P    �  �  �  �  �  �  o  M  )    �  �  �  Y  -  �  �  �  q  �