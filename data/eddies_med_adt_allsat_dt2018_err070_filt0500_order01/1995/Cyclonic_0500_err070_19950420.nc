CDF       
      obs    K   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�������     ,  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       PTG�     ,  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <e`B     ,      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>k��Q�   max       @FW
=p��     �  !0   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vy��R     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q�           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�          ,  98   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��F   max       <D��     ,  :d   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��)   max       B0W�     ,  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��r   max       B0@     ,  <�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >o��   max       C���     ,  =�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Bz�   max       C�{�     ,  ?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          2     ,  @@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7     ,  Al   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7     ,  B�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       PN��     ,  C�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��>BZ�c   max       ?�
=p��     ,  D�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <e`B     ,  F   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>k��Q�   max       @F1��R     �  GH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vy��R     �  S    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @Q�           �  ^�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�@         ,  _P   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�     ,  `|   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����A   max       ?��!�.H�     p  a�         2      
                              "                                             
            !                              "         1   
                           
            $                                    0   	   	N;�N��O��O�#NN�=�OY��O��wO�U�O���O�o�N��dN#��Nl O�hPO�7N;H�O�%�Nʞ�O9+oO#P�NUe~OO��N�#N��N*CLO��N���O�pZPN��O/�N���O"�N��OPTG�N���N�C�O'� O6v8N�a2NDEVNFl�OF48N��5O��N�=�N���O�!�OӨNԶ.OP�}Oh��N:�EO��O8=�O iNP=�N���O�U�O=�OM��OW�)Np�}O�xO�T�N���O��tOP�O	�kNj��N຾N��O@�O�4�NO��N<�<e`B<D��<t�<t�;�`B;D��;o;o;o�D����o��o���
���
��`B�o�o�#�
�49X�49X�49X�T���T���u��o��C���C����㼛�㼛�㼛�㼣�
���
��9X��9X��j�ě��ě���/��/��/��`B��h��h���o�o�C��\)��P��P��P��P��P��w�'0 Ž0 Ž0 Ž8Q�@��D���D���T���]/�ixսm�h�m�h�u��o�����hs��hs���w����������������������������������������������������������������������������������������������������������
#+/102/+#
�����moz�����������zrqmmm���������������;BNR[bgotrf^WN<><:8;����#,32#
��������������� ���������//<HHRKHE<6/////////;BOPZTOLB<;;;;;;;;;;�������������������������������������������������������
'6CO[mrtbO6��������������������	�������#'/<HINQOIH</#"u����������uuuuuuuuu`amqz}�����zmca[WXZ`�������������������������������QUagga]UQNQQQQQQQQQQyz����������������zy��������������������)5Ndqmd`eliNB5)%�����������������������

��������	
#*/:<></#
				?BN[]ggmnng[NJB=67??��������������������#AUn������{_<#
��������������������DHNTamgaaTTTHCDDDDDD������
��������$)16BOY[``[XSOB60( $���������������������������������������������������������������������������������������� �����������������������������^aejmusttqmcaZ[[[]^^HHTU^anssnbaUJHGHHHH���������������������������������������������������������������
#$()(
����#/<HJUY[[UH</#
]anoxyna`_]]]]]]]]]]')36?A;6)�������

�������������������������O[hroh[RODOOOOOOOOOO����������
#04@GFB>0#
���*0<AIJPRUYZZWRI<5+)*��������������������#/:<86899/,#��������������������##)-0<IKUY\\UG<0+###��������������������MOT[_hksohed][TOLGEM��������������������EHU]aglrvrnjaUSKHGDE&),+)&��������������������qt����������tqqpqqqq�����������������������������������������������������������������������3<<HUUXUNHF<33333333�g�c�^�_�g�s�t�x�u�s�g�g�g�g�g�g�g�g�g�g�U�H�?�<�7�:�<�H�U�^�a�f�j�j�n�v�n�i�a�U��
����������#�/�<�G�H�K�U�S�D�<�/�#������������������	��"�*�"������������˽��������������������������A�@�6�/�1�5�A�N�Z�g�k�s�|�z�v�n�g�Z�N�AùìÖÇ�z�o�f�n�zÇì����������������ù����Ƴƫ�����������$�0�;�;�8�0�$�����پ���������(�4�A�f�s�s�k�M�A�4���àÓÌÈÏØàìù����������������ùìà�Z�Y�M�A�@�=�A�M�Z�f�s�z�}�s�f�`�Z�Z�Z�Z������#�/�1�/�)�#����������a�\�`�a�n�zÄ�z�r�n�a�a�a�a�a�a�a�a�a�a�M�F�E�H�H�C�D�M�Z�s����������������s�MD�D�D�D�D�D�D�D�D�EEEEEEED�D�D�Dӿm�e�`�Z�`�m�y�����y�m�m�m�m�m�m�m�m�m�m�G�;�.������޾��	��"�.�6�>�C�G�R�G��w�t�j�t�y¦¦¨¦¢����������������)�5�B�K�B�3�)������������������������������������������àÞÞÜàìïùÿùôìàààààààà�O�L�T�[�c�h�tāčĚĦĲĳĦĥĚā�t�h�O��������������������������������������������������������������������������������������&�(�,�(������������C�B�6�*�)���*�6�C�L�O�X�\�a�\�X�O�D�C�m�e�`�W�`�m�y�����������������y�m�m�m�m����������	��"�;�G�T�`�p�b�Q�?�;�"��T�G�9�-���׾˾ʾ�	��.�G�T�����}�m�T�Ŀ����������������Ŀѿҿܿݿݿ�޿ݿѿĽݽڽнŽʽнֽݽ߽������߽ݽݽݽݿĿ������Ŀпѿݿ�������������ݿѿĿ�ìèàÜÙÙàìù����������ûùìììì�������g�Q�:�5�D�g���������������������������������������(�,�(�����(�#�����(�5�5�A�A�A�;�5�(�(�(�(�(�(����ߺ�������!�(�)�#�!�������.�&�"����"�#�.�;�C�G�T�`�_�[�T�G�;�.�������ݿؿѿͿѿҿݿ���������������ݿۿ׿տݿ�����ݿݿݿݿݿݿݿݿݿ����
���$�(�*�)�(�%����������h�b�[�Q�R�b�h�tāčĚĦĭĲĦĚčā�t�h�T�H�;�/�+�"�����"�/�1�;�H�T�Y�T�T�T������������������*�7�8�<�:�)������Ɓ�~ƁƎƑƚƧƳ������������ƳƧƚƎƁƁ�׾־ʾʾʾ˾;׾���������׾׾׾׻����������,�@�Y�k�w�~�~�w�f�M���������������������
���#�(��
����������ǭǡǟǔǈǆ�{�z�t�{�{ǈǔǖǡǫǭǭǭǭ�����q�m�j�g�k�m�z���������������������������������������������������������ѿ��������ĿѿٿӿѿĿ����������������������{�y�����������������������������������ɺĺ������������������ɺֺ������ֺܺɼ�����&�4�@�M�Y�f�i�c�Y�P�M�@�4�'����������������������������������������������������������������	�	�	�����������׽����������������������Ľнӽ۽۽ݽνĽ��������׽ݽ�������(�4�A�H�C�4����t�g�[�N�B�N�X�[�h�t¥ �tE�E�E�E�E�E�E�E�FFF$F*F$FFF	E�E�E�E���������������������������������û������������������ûлۻܻ��ܻлл��a�T�M�L�T�V�^�d�m�z���������������z�m�a���������������ùϹٹܹ�������ܹϹù���������&�@�P�Y�o�~���~�r�e�Y�3���������������ùϹܹ�����ܹϹƹù����������������Ľнֽݽ����ݽӽнĽ������������������������
����������������ŭŪŨŪŭŶŹ����������������Źŭŭŭŭ�ʼȼǼʼּ׼���������ּ˼ʼʼʼʻ���ܻлû��������ûл����������M�4�'�����'�0�4�@�L�M�Y�f�k�o�l�f�M�<�;�<�A�H�U�a�j�a�a�U�H�<�<�<�<�<�<�<�<EPEDECE<ECECEPEXE\E^E\EVEPEPEPEPEPEPEPEP l > $ r 0  k L W # W g Y 0 K I X l M  i W 6  = ) \ @ R 3 Q 4 + O b N 3 " | o U @ { M t 5 I ^ ( # E M < 4 Y h 6 , H A X B & P 7 \ U = M ) � P G T M    �  '  L  �    �  �  a  x  ~  �  `  J  �  ^  h  ,  @  �  ^  �  �  �  �  1  D  �  �  �  �  �  [    �  �  �  m  |    �  �  �    R  F  �  �  {  �  �  �  T  1  �  U  {      �  �  �  �  P  �    �  B  1  �    �  �    |  l<D���D���#�
��`B�o�u��h��/�D���C��49X�t��#�
�+�#�
�#�
�C���1���ͼ�j�u��/���㼣�
�����h��1�'@���h�����o��w�aG����������8Q�\)���o���m�h�C��}�C���㽥�T�49X�aG��L�ͽY��0 ŽH�9�y�#�L�ͽ<j�Y���7L�}󶽉7L��aG��]/��\)���-���-���㽏\)��o��t���7L�����F�� Ž�jB�3B! �B�gB;�B�B&+B �DB2�Bd@B�YB��BԍB{�B ��BM�B.MB0W�B�B�B��B�A�N�B�4B'�BB �B+�B� Bs�B��B�	Bs�B�7B&�"B)�kA���B�B:&B�tBB�B��B��A��)BhcA���B�B" *B#�B��B�=Bl_B�sB	�B#��B)�@BK�BCZB%VvB&�gB�B�'B�IB&[*B�^B1ZBo�B^Bh�B��B
h�B-dwB�HB� B%B�B��B �jB�BC"B8^B�B ��B��B�BHKB�$B>�B�-B d<B@^B.>B0@BE�B��BEBB�A���BFEB2KB��B lB+5�B��B?�B]�B�BB�B�`B&��B)��A�|�B�uB�B5�B�-BuB��A��rB@�A�c�B3cB"9B>�B~�B�B�B��BɼB#�.B)��B�PB?B%A�B&�[BE"BC�B��B&�nB�YB?8B�SB��BP}B'�B
�B-ĬB�BpB4�B?3A���AŔpA�V�A�	�A0��A�/�A�gBH,A9WA�)*A?r*A�/�A�t�AB�C�;/Ak��A[� A�ˮA� �A���A��
A݅pAI��AJ��A�F5B e�AnMcA`ϹA`��Ax0rA+�(A}�A��A��vA�+A�� @Z�Ab�xA~�
A}<�A�H�A���A��AԺ�BɇATd@�i$A��B�A�%�A�@�Ax�BA��@0�I@�^�A���A���A"�iA3�zA��7C���Bw7@�9(A��->x?��O>o��A'�2A���A�OA?@��N@�:�A�xC���A���A�~�A���A�A0#�A�sbA̔�B	>LA7o]ĄYA?�@A��CA�+ADC�BqAlT�AZ��A��?A��A��sA�[A��<AH6AIxA�f�B MMAl��A`�A]Ay  A+ A~ݫA�=�A��	A��^A��V@S�eAc �A}9�A}�A��{Aܤ!A���A�_lB�AT��@���A�JB>yA��XA��-AyE4A���@3��@�-�A��A�xtA"��A6��A��>C�{�B��@�!�A��>Bz�?��b>F�oA(WtA��A�y.A�]@���@���A�xtC���         2      
                                "                                                         "                              "         2                                          $                           	         1   	   
                     #   %                           %                                 %   7               5                                       !                                                   #      !                                                #   #                                                            !   7               5                                                                                          #      !                           N;�N�.uO
7�OSQ�N��O,��O��wOݚ�O�2�O��N��dN#��Nl O��oO�7N;H�O�;N�tO9+oO#P�NUe~OO��N�#N��N*CLN�C�N���O�;�PN��O/�N���O"�N���PJ�N���N�C�O��O6v8N�a2NDEVNFl�N׈:N��5N�&XN�=�N���O~��OӨNԶ.O>t�Oh��N:�EO��O�YO iNP=�NaE�N�cO=�O5DOW�)Np�}O�xO�T�NVCHO��tOP�O	�kNj��N຾N��O@�O��NO��N<�  8    S      �  �    �  d  �  �  �  f  �   �  D  Q  h  g  �  l  �  )    c  �  ?  �  �  :  K  ,  �  �  �  &  e  D  	  �  �         f  n  l  �  �      �  �  �    Q  �  E  �  	;    .  W    9  �  �  `  �    �  	y  �  �<e`B<o�D��;ě�;ě�%   ;o:�o:�o��`B��o��o���
�o��`B�o��t��49X�49X�49X�49X�T���T���u��o��9X��C���9X���㼛�㼛�㼣�
�ě���j��9X��j���ͼě���/��/��/��P��h�0 ż��o�,1�C��\)����P��P��P�'�w�'<j�T���0 ŽL�ͽ@��D���D���T���}�ixսm�h�m�h�u��o�����hs������w���������������������������������������������������������������������������������������������������������� 
#*---&#
����moz�����������zrqmmm���������������>BN[afmrpgc[NG>A><:>����
#(0/#
��������������� ���������//<HHRKHE<6/////////;BOPZTOLB<;;;;;;;;;;������������������������������������������������������%*36CJOSWTJC6*�����������������������	�������#'/<HINQOIH</#"u����������uuuuuuuuu`amqz}�����zmca[WXZ`�������������������������������QUagga]UQNQQQQQQQQQQ����������������������������������������)5N\kh_[hd[NGB5'!)�����������������������

��������	
#*/:<></#
				?BN[]ggmnng[NJB=67??��������������������	#0BUn�����{]<0#
	��������������������DHNTamgaaTTTHCDDDDDD������	�������$)16BOY[``[XSOB60( $���������������������������������������������������������������������������������������� �����������������������������^aejmusttqmcaZ[[[]^^HHTU^anssnbaUJHGHHHH����������������������������������������������������������������
#'('#"
�����#/<HJUY[[UH</#
]anoxyna`_]]]]]]]]]]')36?A;6)������

��������������������������O[hroh[RODOOOOOOOOOO��� ��������#04;;900#*0<AIJPRUYZZWRI<5+)*��������������������#/:<86899/,#��������������������##)-0<IKUY\\UG<0+###��������������������JOY[`hlhh[OKJJJJJJJJ��������������������EHU]aglrvrnjaUSKHGDE&),+)&��������������������qt����������tqqpqqqq������������������������������������������������������������������������3<<HUUXUNHF<33333333�g�c�^�_�g�s�t�x�u�s�g�g�g�g�g�g�g�g�g�g�U�P�H�B�<�H�J�U�a�c�d�h�b�a�U�U�U�U�U�U���
���
���#�/�1�9�<�=�<�:�/�#������������������	���"���	���������׽�������������
�����������������N�D�A�:�3�5�7�A�N�Z�g�s�w�w�s�s�j�g�Z�NùìÖÇ�z�o�f�n�zÇì����������������ù����ƴ�������������$�0�:�:�7�0�$����پ��� �	���(�4�A�^�f�s�h�Z�M�A�4���àÓÏÍÏÒÛàìù��������������ùìà�Z�Y�M�A�@�=�A�M�Z�f�s�z�}�s�f�`�Z�Z�Z�Z������#�/�1�/�)�#����������a�\�`�a�n�zÄ�z�r�n�a�a�a�a�a�a�a�a�a�a�M�I�K�K�G�I�M�Z�s����������������s�Z�MD�D�D�D�D�D�D�D�D�EEEEEEED�D�D�Dӿm�e�`�Z�`�m�y�����y�m�m�m�m�m�m�m�m�m�m���	��������������	��"�(�-�-�"�!��y¥¦§¦ ����������������)�5�B�K�B�3�)������������������������������������������àÞÞÜàìïùÿùôìàààààààà�O�L�T�[�c�h�tāčĚĦĲĳĦĥĚā�t�h�O��������������������������������������������������������������������������������������&�(�,�(������������*�%�#�*�6�C�M�O�C�6�*�*�*�*�*�*�*�*�*�*�m�e�`�W�`�m�y�����������������y�m�m�m�m���������	��"�;�T�`�e�i�`�^�N�;�.�"����T�G�9�-���׾˾ʾ�	��.�G�T�����}�m�T�Ŀ����������������Ŀѿҿܿݿݿ�޿ݿѿĽݽڽнŽʽнֽݽ߽������߽ݽݽݽݿĿ������Ŀпѿݿ�������������ݿѿĿ�àßÛÞàìù����������ùìàààààà�������v�g�S�>�I�g���������������������������������������(�,�(�����(�#�����(�5�5�A�A�A�;�5�(�(�(�(�(�(�������������!�&�'�"�������.�&�"����"�#�.�;�C�G�T�`�_�[�T�G�;�.�������ݿؿѿͿѿҿݿ���������������ݿۿ׿տݿ�����ݿݿݿݿݿݿݿݿݿ����
���$�(�*�)�(�%����������t�p�h�]�[�[�[�h�tāčĚĘčĂā�t�t�t�t�T�H�;�/�+�"�����"�/�1�;�H�T�Y�T�T�T�����������������)�,�/�-�)���Ɓ�~ƁƎƑƚƧƳ������������ƳƧƚƎƁƁ�׾־ʾʾʾ˾;׾���������׾׾׾׼���&�4�@�M�Y�f�o�v�u�n�f�Y�M�@�4�'������������������
���#�(��
����������ǭǡǟǔǈǆ�{�z�t�{�{ǈǔǖǡǫǭǭǭǭ�����z�s�m�h�m�z�����������������������������������������������������������ѿ��������ĿѿٿӿѿĿ����������������������{�y�����������������������������������ɺ������������������ɺֺ�����ֺκɼ�����&�4�@�M�Y�f�i�c�Y�P�M�@�4�'�������������������������������������������������������������� �����������������佞�����������������ĽĽƽĽ��������������������׽ݽ�������(�4�A�H�C�4����w�t�g�b�`�g�o�t E�E�E�E�E�E�E�E�FFF$F*F$FFF	E�E�E�E���������������������������������û������������������ûлۻܻ��ܻлл��a�T�M�L�T�V�^�d�m�z���������������z�m�a�ù����������ùϹϹܹعϹùùùùùùùú�������&�@�P�Y�o�~���~�r�e�Y�3���������������ùϹܹ�����ܹϹƹù����������������Ľнֽݽ����ݽӽнĽ������������������������
����������������ŭŪŨŪŭŶŹ����������������Źŭŭŭŭ�ʼȼǼʼּ׼���������ּ˼ʼʼʼʻ���ܻлû��������ûл����������'�� �'�0�4�@�M�Y�f�k�m�n�k�f�Y�M�@�4�'�<�;�<�A�H�U�a�j�a�a�U�H�<�<�<�<�<�<�<�<EPEDECE<ECECEPEXE\E^E\EVEPEPEPEPEPEPEPEP l =  n /  k I U & W g Y ( K I . V M  i W 6  = ( \ E R 3 Q 4 * L b N 0 " | o U 2 { ( t 5 4 ^ ( # E M < % Y h #  H & X B & P ( \ U = M ) � P D T MBq  �  �  $  :  �  j  �  #  9  %  �  `  J  G  ^  h  O  �  �  ^  �  �  �  �  1    �  �  �  �  �  [  �  �  �  �  W  |    �  �  �    �  F  �  �  {  �  �  �  T  1  !  U  {  e  �  �  '  �  �  P  �  f  �  B  1  �    �  �  D  |  l  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  8  1  +  %            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �       �  �  �  �  �  �  _  :    �    $  �  �  9  z  �  �    0  H  S  E  %  �  �  �  =  �  ;  �  �    �            �  �  �  �  z  T  (  �  �  �  |  P    �  �            �  �  �  �  �  �  �  �  }  g  N  6      �  �  �  �  �  �  �  �  �  s  Z  ?  !     �  �  K  �  �    �  �  �  f  L  5       �  �  �  �  �  [  0  �  �  @  �  [  �  
    �  �  �  �  �  �  m  =  	  �  �  p    �    u   �  �  �  �  �  �  �  �  �  �  �  �  �  y  d  K  +     �  �  �  Q  X  a  b  Y  N  E  <  /      �  �  �  Y  �  �  N  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  k  ]  K  8  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  m  _  Q  B  4  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  r  l  f  Z  c  f  c  Y  E  -    �  �  �  �  �  e  8  	  �  �  x  �  �  �  �  �  �  �  h  B    �  �  ,  �  Z  �  [  �  <  �  �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �    !  +  5  =  C  B  <  .    �  �  �  �  H  �     o     /  M  @  .    �  �  �  �  s  K  $  -  9  @  G  O  X  a  h  \  R  J  C  ;  /      �  �  �  �  �  }  L    �  �  �  g  f  e  e  a  Z  M  <  (    �  �  �  �  �  a  D  @  o  �  �  �  �  �  �  �  �  �  �  {  b  <    �  �  �  �  d  C  !  l  U  @  ,      �  �  �  �  �  �  d  3  �  �  �  `  +  �  �  �  �  �  �  �  �  �  �  �  �  �    u  c  Q  >  (    �  )  !      
    �  �  �  �  �  �  �  �  �  �  �  �  v  k    	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  \  [  Y  Z  [  _  a  b  [  T  M  E  :  ,      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  w  r  f  Y  L  >  1  $  8  >  >  8  .      �  �  �  �  �  f  +  �  �    �  m  �  �  �  �  �  �  �  �  n  B    �  �    R  $      �   �  �  �  �  �  �  �    i  R  :  #    �  �  �  �  �  h  r  �  :  -        �  �     	     �  �  �  �  v  a  K  4      K  E  >  6  *    	  �  �  �  �  �  z  Z  2  	  �  �  �  b  �      %  ,  (  "      �  �  �  t  E    �  �  u  2  �  �  �  �  �  �  e  ?    �  �  �  �  i  S  <  "  �  �  b   �  �  �  �  �  �  �  �  �  �  �  �  |  o  b  U  E  5  %      �  �  �  �  �  �  �  �  z  q  i  `  W  N  D  ;  1  '      &  &  %        �  �  �  �  a  '  �  �  [    �  �  *  �  e  e  d  b  _  Y  S  N  E  ;  ,    	  �  �  �  �  Q     �  D  B  ?  <  :  5  *        �  �  �  �  �  �  �  n  S  8  	    �  �  �  �  �  �  �  �  �  �  �    r  d  Y  O  E  <  �  �  �  �  �  �  ~  v  n  f  ^  U  M  D  <  4  -  %      �  �  -  j  �  �  �  �  �  y  _  B    �  �  q     �  �  �    �  �  �  �  �  �  �  �  q  ^  K  6     
  �  �  �  W     �  �  �  �  �  �  �  �       �  �  �  -  �  :  �  �  H            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  Y  M  @  /    
  �  �  �  �  �  �  v  ]  D  *     �   �  1  R  c  j  m  g  S  0    �  �  c    �  A  �  r  �     �  l  a  V  O  G  =  1  #        �  �  �  �  e  !  �  �  W  �  �  �  �  }  d  G  $  �  �  �  �  [  -  �  �  U    �  `  �  �  �  �  �  �  �  �  q  _  I  .    �  �  �  �  �  s  M        �  �  �  �  �  �  �  g  C    �  �  �  O    �  h         �  �  �  �  �  �  �  g  H  *    �  �  �  T     �  �  �  �  l  P  3    �  �  �  �  a  /  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  M    �  �  `  (  �  D  �  �  �  �  �  �  �  {  l  \  I  .    �  �  �  \  &   �   �      �  �  �  �  �  �  �  �  m  V  @  +       �   �   �   �  >  D  J  M  O  P  P  M  I  F  =  -    �  �  �  z  M     �  R  O  J  P  f  �  �  �  �  �  �  t  ^  ;    �  �  J  �  �  E  ?  .              �  �  �  y  G    �  �  &  �  �  d  �  �  �  �  �  �  �  �  �  U  !  �  �  P     �  �  ^    	;  	6  	.  	  �  �  �  �  �  a  �  v  �  $  �  C    �  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  ~  j  T  >  .  %        �  �  �  �  �  �  �  t  g  \  R  G  :  -  !  W  <  #    �  �  �  o  D    �  �  �  z  N       �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  #  �  4  �  )  9  &  
  �  �  �  �  �  �  �  �  n  \  2  �  �  P  �  �  �  �  �  {  k  Z  G  /    �  �  �  f  1  �  �  �  E  �  i  �  �  �  �  �  z  l  Z  B  (    �  �  �  �  ~  d  O  J  |  �  `  X  O  F  >  6  /  (  !            �    	        �  x  p  h  `  V  E  5  !    �  �  �  �  �  �  �  �  �  r    
       �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  k  �  �  �  u  ?    �  �  9  �  �  V    �  {     �  A  �    	q  	u  	c  	J  	+  	  �  �  r  )  �  s    �  +  �  2  �  �  �  �  �  �  �  �  ~  o  a  R  B  2       �  �  �  �  |    �  �  e  C    �  �  �  �  e  C  !  �  �  �  z  B  �  �  
   �