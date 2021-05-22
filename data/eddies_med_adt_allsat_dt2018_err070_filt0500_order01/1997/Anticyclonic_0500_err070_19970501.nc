CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�XbM��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �t�   max       =�^5      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F��Q�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��    max       @vvfffff     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @P�           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�F        max       @��           �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >t�j      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�{�   max       B3	�      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�z�   max       B3|�      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?V�   max       C�y�      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?P+�   max       C�w       �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P\��      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�X�e,   max       ?� [�6�      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       =��m      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F��Q�     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @vvfffff     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P�           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�F        max       @��@          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��,<���   max       ?�����>C     �  V�               �            	      *      (            9                     D                           "   +   .         1   	   	   0         	               8   >   )      �      
            "            DN�׷N��UN���N�t�P���N�E�O'[N?Z�N�V�O.��OM׹NV��O`6N�P�O29VN��O��N��O)�JN"~Ojb�N�?N�_AP*N��NN��OZ�O�KNJW�NO���N�x�O�w�O�]�O���N�K�Ojz@O�diO)�hN���P# NZ#OO��bO��ND߫N���NX]CN�j�P\��Pw�O�@�N�{P�vXN��NO0-1O��N�@�N�DYO��4N痾O%��N��O.��t���1���㼋C��e`B�D���49X�49X�t���`B��`B��o�o�o%   :�o;o<o<t�<#�
<D��<e`B<�o<�t�<�t�<���<��
<�1<�1<�9X<�j<���<���<�/<�/<�/<�h<��<��<��=o=o=t�=t�=�P=�w=#�
=#�
=#�
=#�
='�='�=,1=,1=49X=H�9=L��=]/=q��=�%=�O�=�Q�=�^5NW[gqt}xtgb[NNNNNNNN���������������������vuty���������������w{������������wwwwww����)Ngxzv[MB5
���
#/0840#
YROV[gt��������tg_[Y��������������������]bhkt������~th]]]]]]������������������������������������������������������������	#/<EHG@</#
	��������������������ecnz|������������zne��������������������plou���������������p���������� ������������ )5=<54-)�#'($#����������������������������������������.))/4<FHUUULLH</....303?Oht��������t[L<3��������������������#/3:6/,#�����
#)-3/#
��������)()(+5BDB)����������������������"%////" �)5>@HKQNB52 
 ��������������������	
#/<HQU`b`UH<#	��������������������JO[h�����������th[OJ����������������������������������������#<FR[[UW]UH?<>??BO[chporqjh`[OIB>0.0<IJUbecbXUI=<<000���)BNSTNB1)
��������������������������.B[``NB5)��y{���������������{y 
#$'$# 
        ��������������NKLTTabhca[TNNNNNNNNdheddkmpz���zqmdddd������5K[jg[B����������������������������������������������������������������-1.	�������behou}��������uhbbbb���������������������������

������������������������������������������������%)& �����������������������������&)41)%���������������������������������	
 �������zÇËÓÇÄ�z�n�f�m�n�o�z�z�z�z�z�z�z�z�0�=�I�O�T�Q�I�=�9�0�/�0�0�0�0�0�0�0�0�0ĳĿ��������������������ĿĶĳĳĳĳĳĳ���
������
��������������������������<�R�^�j�a�S�T�K�0�#������ļħ������������������������x�t�p�u�x�����������������������������������������������������Ż����
�	������������������@�L�Q�Y�^�\�Y�L�@�3�2�3�7�5�@�@�@�@�@�@�������������
��������������ƿ�������̼������ʼּ����ּμʼ�������������������������������������������������������EEEEEEED�D�D�D�D�D�D�D�D�D�D�EE�����������������������E�E�E�E�E�E�FFFFFE�E�E�E�E�E�E�E�E���������������������������������������6�B�O�[�k�q�q�h�B�:�)�"�����������6�����þʾվʾȾ����������������������������)�6�<�A�F�J�J�B�6�,�)��������H�T�a�g�f�a�T�H�?�A�H�H�H�H�H�H�H�H�H�H�������������������������żŻ���ƿ�������������������������������|�������f�s���������������~�s�f�`�[�b�f�f�f�f�����ʾ׾پ;���������f�Z�P�K�I�M�Z�s��àåìòùýùùìàÓÎÓÕÙÕàààà�������������������������������t������������������������������������������y���������������������r�`�G�A�@�R�`�m�y�Z�f�j�f�c�f�f�f�Z�M�M�C�M�P�Z�Z�Z�Z�Z�Z�T�Z�a�a�a�U�T�H�D�C�H�L�T�T�T�T�T�T�T�T�M�Z�f�q�q�p�f�Z�A�4�(�%����(�-�7�A�M���������������������������������������ҿT�`�m�}�������}�p�m�^�T�G�;�1�/�2�<�G�T�	��"�-�4�8�6�/�"��	�����������������	�����Ƽռ�ۼּʼ���������������������������	�����	��������������𼽼ʼּ��ټʼ��������~����������������������������������ùåÛìî�������#�/�<�H�Q�^�a�e�a�U�H�<�/�#������#�����������������������}�z�r�r�r���ƧƳ��������� ������ƦƘƁ�h�V�\�d�uƏƧ�<�H�U�W�U�P�H�<�<�/�/�,�/�0�<�<�<�<�<�<�(�5�A�O�R�L�@�(�����޿ڿؿ޿����(�Ľ˽нԽǽĽ��������z�y�q�y�����������ļ��� ��������������������m�m�a�`�T�G�C�;�.�"��"�&�.�;�F�G�T�`�m�b�n�{ŇŇŊŇ�{�n�j�b�\�b�b�b�b�b�b�b�bŔŠŭŹ������������ŹŭŠřŔŏŔŔŔŔ�������	�;�T�`�U�S�D�"������������������N�s�����������������s�Z�N�5�!����(�NāčđĚĜĘĔćā�t�[�O�B�1�6�B�O�[�hā���������	��"�+�"���	���������������׼'�@�M�`�k�n�m�e�@�'����������û���'��������	��	�������޾ݾݾ����'�3�6�D�C�@�3�'����������������'��(�5�5�A�G�L�G�A�5�(��������������������������������{�z���������������`�j�l�u�y�{�y�y�s�l�`�Z�S�R�Q�N�S�T�`�`���
��%�.�#���������¿·½¿�����������������!�������������������������-�:�F�O�S�^�\�X�S�M�F�:�4�-�)�!��!�$�-ù������������ùìàÓÍÓàìïùùùùDoD{D�D�D�D�D�D�D�D�D�D�D�D{DwDoDkDcDcDo T ; & C 5 V H ? A 3 4 h % : Z g 2 , N K 8 . ' 6 U V X ] @ 5 E < J > 2 A B K > d > C l c [  F R H 8 < | X S J - - \ [ " 3 � R    �  �  �  �  �  �  }  p  �  �  �  �  �  �  �  X  Y  �  �  6  �    �    �  �     �  d  ;  K  '  �  c  m  �  �  �  �  �  �  �  �  �  �     }      �  9  O  �  �  �  <  �    x  �  s  �  u�o�u�o�49X>$ݻ��
:�o��o%   <T��=\)%   =�P<49X<�t�;�o=m�h<e`B=C�<T��=+<���<�9X=� �<�=+=C�=,1<�h<���=�w=t�=q��=�\)=�t�=C�=Y�=���=#�
=�w=���=��=y�#=49X='�=<j=<j=<j=ě�=��`=��=D��>t�j=<j=]/=�C�=}�=��=�j=��=�E�=��>!��B	)B��B
x�B
��B!PB%2aB	�%B �=B)gB�BP�B�B��B �B��B�zBa�B7B0B��B�UB�{B�}B�2B!��B�lB�BD3B<�A�{�Bb�B9�BUB��BlEB!�B"z�B��B%�B&�_Bq	B�3B{�B)��B$��B@�A���A���BeFB�B~YB�B/�B3	�BFB#�B ��B-M2BOBE�B,�B�~B�VB	�
B�B
~�B
�sB�"B%7�B	��B D8BD�BȸB@B
�]B�sB 1uBABIB~BBCB=@BM�B�mB��B�iB��B">B��Bb$B��B@.A�z�BP�B�-B=�B��B��B!BSB"DB��BB&rGB;oB�KBP�B)�KB$�ZB?�A�|�A�r�B��B�4BrPB?�B@B3|�B@*BA'B �pB-��B>�B:�B��B��B�A�~B
��A㳽A��A�<@�G�A�]�@��	?���B9@�^-A�+�C�F3?V�C�y�A��mA��AMv[Aֶ�A���A�fiArC5AC�MAF�WA˾�A�_>A�r�Al��A>��A�?�A>�DA�>{Ag��A��@��0AYU@�2%A�hA�w@�	B��AþdA��A"m
A+�Ads�A��A��#A��$A��A�A��)@�h�AV��?�wA��A��AL*A��A�ih@~��A�C�C���A�_B�A�z�A��&A��@�~A��|@���?��Bc%@�OA��C�H�?P+�C�w A��<A���AMXAց_A��A�<�Ar�	AC��AF�sA�r�A���A���Al��A>�{A�8A?A�~RAh�'A�xD@�yrAY$@�#�A�X�AãX@�@�B��AÀ�A���A!XA�Ad�A��A�s�A�zjA��A�y A�f@�GAX)�?P[�A�BA ��A��A��RA�j�@|7�A�bC���               �            
      +      (            9                     E               	            "   ,   .         1   
   	   0         	               9   ?   *      �                  "            D               ;                                    '                     +            '                     !         #         -      )                  1   )         9                                             %                                                         !                                                   #      )                  1   !         #                              N�׷NAN���N�t�P	�Nb7�O'[N?Z�N�V�O
AO�NV��O�XN>цO29VN��O ��N��N޹^N"~OXpuN�?N�_AO�0�N��NN��O�-O�Q�NJW�NN��-N�x�O�w�O���O8g1N�K�Oh�N�3 O)�hN���O��NZ#OO���N��END߫N���NX]CN�j�P\��O��OG�-N�{O�|�N��NO0-1N�2N�@�N�DYO5oN痾O�+N��O*/�  �  �  �  �  q    �  ?  x  E  	     �  |  q  ]  $  �  �    �    W  �  �  #  �  �  �  �    �  }  �    8    [  2    G  4  �  �  z  T  �  �  l  �  �    �  �  P  O  �  	  �  m    �  ǽt����
���㼋C�=8Q�49X�49X�49X�t���o;o��o;�`B;D��%   :�o<��<o<u<#�
<T��<e`B<�o=��<�t�<���<�j<�j<�1<�9X<��<���<���<�=#�
<�/=t�=e`B<��<��=0 �=o=��=�P=�P=�w=#�
=#�
=#�
=e`B=D��='�=��m=,1=49X=L��=L��=]/=�C�=�%=�\)=�Q�=�jNW[gqt}xtgb[NNNNNNNN���������������������vuty���������������w{������������wwwwww������5ISP?5) ��#,0620#YROV[gt��������tg_[Y��������������������]bhkt������~th]]]]]]������������������������������������������������������������#*/<>CB<:/#��������������������ecnz|������������zne�������������������������������������������������� ���������!)/5750))#'($#����������������������������������������.))/4<FHUUULLH</....B?>BO[hr�������th[QB��������������������#/3:6/,#�����
!#%&
��������$'&)/@@=5)����������������������"%////"
),5653)





��������������������	
#/<HQU`b`UH<#	��������������������WY_ht���������tqh^[W����������������������������������������*$%(/<BHLJHA</******>??BO[chporqjh`[OIB>0.0<IJUbecbXUI=<<000���)5BNNJC6+)��������������������������-BX\YNB5)��{�����������������{{ 
#$'$# 
        ��������������NKLTTabhca[TNNNNNNNNdheddkmpz���zqmdddd������5K[jg[B���������������������������������������������������������������������behou}��������uhbbbb��������������������������

������������������������������������������������������������������������������$)/+)#��������������������������������
 �������zÇËÓÇÄ�z�n�f�m�n�o�z�z�z�z�z�z�z�z�=�I�M�R�N�I�=�:�0�0�0�9�=�=�=�=�=�=�=�=ĳĿ��������������������ĿĶĳĳĳĳĳĳ���
������
����������������������������#�0�<�B�I�J�D�7�0�#�
���������������x���������������x�u�r�w�x�x�x�x�x�x�x�x���������������������������������������Ż����
�	������������������@�L�Q�Y�^�\�Y�L�@�3�2�3�7�5�@�@�@�@�@�@�������������������������������������ټ����ʼּ޼���ּʼ�����������������������������������������������������������D�D�D�EEEEEEEED�D�D�D�D�D�D�D�D߹�������������������������������E�E�E�E�E�E�FFFFFE�E�E�E�E�E�E�E�E���������������������������������������6�B�O�[�^�]�[�T�O�B�A�6�)������)�6�����þʾվʾȾ���������������������������)�.�6�=�A�B�D�B�=�6�3�)��������H�T�a�g�f�a�T�H�?�A�H�H�H�H�H�H�H�H�H�H�������������������������Žż���ҿ�������������������������������|�������f�s���������������~�s�f�`�[�b�f�f�f�f�����������������������s�a�X�U�V�\�f�àåìòùýùùìàÓÎÓÕÙÕàààà�������������������������������t����������������	����������������������������������������������y�`�T�G�D�B�T�`�m���Z�f�j�f�c�f�f�f�Z�M�M�C�M�P�Z�Z�Z�Z�Z�Z�T�Z�a�a�a�U�T�H�D�C�H�L�T�T�T�T�T�T�T�T�Z�f�g�i�h�f�\�Z�T�M�C�E�M�Q�Z�Z�Z�Z�Z�Z���������������������������������������ҿT�`�m�}�������}�p�m�^�T�G�;�1�/�2�<�G�T�	���"�+�2�6�5�/�"��	���������������	�������ǼϼҼʼƼ���������������������������	�����	��������������𼱼��ʼмּۼּռ̼ʼ����������������������������	����������������������������#�/�<�H�Q�^�a�e�a�U�H�<�/�#������#�����������������������}�z�r�r�r���ƧƳ����������������ƳƧƚƎƉƀ�ƋƚƧ�<�H�U�W�U�P�H�<�<�/�/�,�/�0�<�<�<�<�<�<�(�5�A�M�P�I�>�(�������ܿۿ�����(�ƽнҽŽĽ������������y�������������½Ƽ��� ��������������������m�m�a�`�T�G�C�;�.�"��"�&�.�;�F�G�T�`�m�b�n�{ŇŇŊŇ�{�n�j�b�\�b�b�b�b�b�b�b�bŔŠŭŹ������������ŹŭŠřŔŏŔŔŔŔ�������	�;�T�`�U�S�D�"������������������s�������������g�Z�N�?�5�(�#�!�*�A�N�Z�s�h�tāčĕĔĐčĄā�t�h�[�O�B�L�O�[�]�h���������	��"�+�"���	���������������׼�'�4�@�M�R�W�Y�Y�S�M�@�4�'����������������	��	�������޾ݾݾ����'�3�6�D�C�@�3�'����������������'�(�.�5�A�F�L�F�A�5�(������ �(�(�(�(�����������������������{�z���������������`�j�l�u�y�{�y�y�s�l�`�Z�S�R�Q�N�S�T�`�`�������
�� ����
���������������������������!�������������������������-�:�F�M�S�]�[�W�S�J�F�:�8�-�*�%� �!�%�-ù������������ùìàÓÍÓàìïùùùùDoD{D�D�D�D�D�D�D�D�D�D�D�D{DwDoDkDdDdDo T . & C $ F H ? A 1 $ h & 9 Z g = , E K ; . ' , U V T X @ 5 < < J 6 # A .  > d 6 C k k [  F R H 4 7 | $ S J  - \ F " . � R    �  ^  �  �  b  n  }  p  �  %  %  �  E  T  �  X  b  �    6  �    �  �  �  �  r  �  d  ;  �  '  �  '  �  �  !  �  �  �  �  �  �  '  �     }      �  �  O  �  �  �    �    �  �  Q  �  o  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �  t  d  U  E  5  &  #  ,  6  @  I  H  C  >  9  4  �  �  �  �  �  �  �  �  �  v  c  I  /    �  �  �  �  �  i  �  �  �  �  �  �  �  �  l  X  D  0    
  �  �  �  �  �  ;  �  �  �  �  �  �  �  �  �  v  `  F  -    �  �  �  }  I    ^  �  	g  
  
�    F  f  p  h  N    
�  
�  
  	G  _  +  [  �        ~  z  v  o  f  ^  Q  C  3    �  �  �  �  f  L  2  �  �  �  �  �  �  �  �  �    v  m  \  H  1    �  �  �  X  ?  ?  ?  9  %      �  �  �  �  �  s  J     �  �  �  g  8  x  k  ^  T  L  H  F  A  ;  1  #    �  �  �  �  ^  '  �  �    2  D  D  ?  7  ,      �  �  �  \  (  �  �  q    �  #  p  �  �     	    �  �  �  �  �  R    �    K  p  �  a   �           �  �  �  �  �  �  z  Z  :     �   �   �   �   �   �  c  �  �  �  �  �  �  �  �  z  O    �  p    �  .  �  d  �  5  N  a  m  t  x  {  |  z  v  l  Z  D  (    �  �  �  \  /  q  I    �  �  �  �  �  �  �  �  |  h  S  9    �  �  x  )  ]  O  A  3  %    	       �  �  �  �  �  �  �  �  �  �  �  �  �  .  Q  t  �  �  �  	    $      �  �  P  �  +  N  {  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  b  M  7  "  0  P  k  }  �  �  |  U  -    �  �  U    �  S  �  M    �                    �  �  �  �  �  �  �  w  T  0    �  �  �  �  �  �  �  �  |  e  K  +    �  z  '  �  �  1  �      �  �  �  �  �  �  �  �  |  r  i  a  X  P  E  :  /  #  W  S  P  L  F  @  ;  1  %        �  �  �  �  �  �  �  �  �  3  b  �  �  �  �  �  �  w  ]  9    �  a  �  �  �  �  )  �  �  �  �  �  �  �  x  ^  X  T  T  T  V  \  c  k  v  �  �  #    	    �  �  �  �  �  �  �  {  P  "  �  �  �  �  �  %  �  �  �  �  �  �  �  �  �  �  �  �  �  r  ]  G  0    �  �  �  �  �  �  �  �  �  �  t  X  5  
  �  �  S  �  �  6  �  q  �  �  �  �    t  l  d  `  ]  U  H  <  0  %      �  �  �  �  �  �  �  �  �  n  U  =  $    �  �  �  x  S  -     �   �  s  �  �  �  �  �  �  �  
        �  �  �  s  5  �  �  6  �  �  �  �  �  �  �  �  �  �  �  �  w  T  0    �  �  �  R  }  t  m  \  G  .    �  �  z  7  �  �  :  �  �  /  �  )  ~  �  �  �  �  �  �  �  m  D    �  �  b    �  #  �  �  �  R  �  �  �  �  �         �  �  �  �  s  3  �  �  0  �  �  ?  8  1  *  $          �  �  �  �  �  �  �  u  d  U  F  7  �  �  �            	  �  �  �  �  �  �  ^    �  6  �  �  �  9  e  }  �  
  1  F  T  Y  M  5  �  �  +  �    I  f  2  *  "      �  �  �  �  �  o  R  4    �  �  �  f     �        �  �  �  �  �  �  �  �  l  V  ?  (    �  �         *  +  '  A  C  7  )    �  �  �  g  	  �    t    �  
  4  /  +  &          �  �  �  �  �  �  �  e  G  '    �  �  �  �  �  �  �  �  s  T  /    �  �  >  �  �  J  �  G  e  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  e  L  3    z  r  j  b  [  R  E  8  +          �  �  �  �  �  �  �  T  L  D  A  H  N  G  6  %    �  �  �  �  �  y  g  T  A  .  �  �  �  �  �  �  �  �  �    n  Z  F  3      �  �  #   �  �  �  �  �  �  �  �  �  w  h  X  I  9  )      �    >  ^  l  Z  j  ^  I  :  -     *    �  �  �  V    �  P  �  _  t  p  �  �  �  �  �  �  �  �  }  D  �  �  C  �    R  �  9  �  �  �  �  �  �  �  �  �  �  Z  &  �  �  d    �  �  <  �  �    �  �  �  �  �  j  H  %    	  �  �  �  �  �  �  c  ;    �  Q  U    �    J  w  �  k  A  �  �    H  J    
�    o  �  �  �  �  �  �  �  �  ~  w  o  h  `  X  P  L  I  F  B  ?  P  C  4      �  �  �  n  :    �  �  �  �  x  U  2     �  M  O  K  B  4  &    	  �  �  �  �  k    �  $  �  >  �  D  �  �  �  �  �  �  q  Y  ?  "    �  �  �  Y  "  �  �  _  "  	  �  �  �  �  �  Z  1    �  �  �  �  h  L  -    �  �  �  �  �  �  �  �  �  �  �  �  s  C  	  �  �  @  �  �  3  �  4  m  f  ]  Q  A  -    �  �  �  n  =    �  �  o  >    �  �        �  �  �  �  �  v  U  0     �  w  &  �  �  @    �  �  �  |  ^  A  $  	  �  �  �  �  �  �  �  �  �  �  �    |  �  |    �  v  B    �  T  �  Z  �  <  �  �  
�  	~  /  �  �