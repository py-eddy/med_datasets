CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�\(��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M˼�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��G�   max       <�1      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��z�H   max       @E�Q��     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v:�\(��     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @O@           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�`          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <e`B      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B*�*      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~   max       B+6�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�7   max       C���      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Fm�   max       C��      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          V      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M˼�   max       P�\      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�PH   max       ?�]�c�A!      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��G�   max       <��
      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��z�H   max       @E�Q��     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �θQ�     max       @v:�\(��     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @N�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�'`          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Ag   max         Ag      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?!-w1��   max       ?�Z���ݘ     �  T      #                     U      4      .               &            J   K   +                     
            B   	                                           !   C            #               N��O��mO�[iN%��N��N�d�N�2	N0�mP���N���Pr�Nk<�O�ҎO!�-N���N�C:Op(0O�2�N"��N��M��PYjP��Og��OQ��O+ mO0߃N�8�N��O��N�^=O�x�O���N�bP?��N��M˼�O���Nqk{O��CO)�
O2�O�y�O���O�N��N��OL�"N9��O_�4P��O�`sNN��OS�O��N���O�(Nbn�N��yO�\<�1<�o<D��<o;D��:�o:�o:�o�o�o��`B�t��t��t��49X�e`B�u��o��C���t���t���1��1��h�������o�+���#�
�#�
�#�
�'0 Ž0 Ž0 Ž49X�49X�49X�D���D���P�`�P�`�P�`�P�`�ixսixսm�h�m�h��o���������-��E���Q콸Q콺^5�����G���������������������Uaz�����������sa\WUUXanz����������za^]YX"#,/<HMHF</#"""""""")16A66)(����������������������������������������9<HLMJH<9:9999999999�����
-87. �������pz����������zwpppppp?O[ht�����������hC??ST\alhedaTRQSSSSSSSStz���������������npt')5BDLNQSNCB5)#/<B<30/%#�������������������� #/:<@DHD?</#������

������������������������������Q[]cghjg[ZYTQQQQQQQQdgttt{���tqgddddddddz~����������������~z��%/84)�������}������������������}��������������������	#/4:<=<6/# 	bhn{���������{xnba]b� ���������/5BDLNONLB>5/*//////jmpz��������zmhgjjjjQTUamtwxxmmmaVTQQQQQ�������������������������������������������������������������5AGFBH[NB) ����it�����������wtmiiii))/.6897765)))))))))������������������������������������,9@HU^n{�����{UK><0,INP[gt��������tg[ONI��������������������)5[fi]Z[]TB5'$$tz}�����������xtmpst�������
"�����������������������������AIU`bfglbUKIF>AAAAAAv{������������{sonv��������������������M[htsrpormhf`[VOGDDMAENg�������������[NA��������������||���[[hhpnh[PQ[[[[[[[[[[nnz���������zunnjknn<?HUUaz��zqaUHCBB<</009<CILLKII?<<:50///017;<DIKOTTQI<40/./<?<;/%##/<<<<<<<<<������������������������������������������������������������������������������������������6�C�O�C�*��������s�i�_�\�b�g�x�|�����������������������s����������������������������������������ɺȺźɺϺֺ���������ֺɺɺɺɺɺ�����������������������������������������¦¥¦²¸¾²­¦¦¦¦¦¦¦¦������(�.�1�(��������������g�Y�A�#�*�5�N�g�������������������������������������������������������������һx�M�>�A�_�k�>�;�S�_�x���ûܻ׻��Ȼ����x�g�e�Z�U�Z�g�s���������s�g�g�g�g�g�g�g�g�5�,�����ݿݿ�����5�A�T�a�h�e�N�A�5�5�3�(�'�"�&�(�5�=�A�N�R�Z�[�Z�W�X�N�A�5ìèâäéìíù��������������ùìììì�����}�����������������������������������/�#���#�%�/�<�U�a�n�t�y�x�n�a�U�H�<�/�&��$�B�F�T�V�Y�f�w�w�~����������f�M�&�������)�.�)�#�����������������������ĿʿƿĿ��������������������ݿտѿѿѿݿ��������ݿݿݿݿݿݿݿݻлû��x�_�F�3�0�5�C�S�����ܻ������л�������'�M�f�w�~��������r�M�'���Ŀÿ��������Ŀѿݿ���������޿ѿſ����������������
���#�/�,�#��
���������O�B�6�-�/�2�6�B�H�O�[�h�r�r�t�x�t�h�[�O�����������������ʼϼּؼټؼּм˼ʼ����ʾľȾʾ׾������ �����׾ʾʾʾʾʾ��a�\�_�a�m�x�z�~���������z�m�a�a�a�a�a�a�)� �����)�)�6�?�B�K�M�K�B�6�)�)�)�)��ƴƳƭƮƳ���������������������������������������������$�0�9�7�1�$��������m�V�P�T�`�y���������Ŀ޿������Ŀ����mŹŷůŭŬŬŭŹ��������������żŹŹŹŹÇ�r�a�M�/�<�H�nÓìù��������������ìÇùôóìççìòù��������������ùùùù��������þ�������������������������������{�r�b�]�T�M�I�>�I�V�o�{ǍǖǖǔǒǑǈ�{ÓÇÇ�~ÆÇÓÚàäèàÓÓÓÓÓÓÓÓ���������g�Z�L�I�N�S�g�s�����������������������߾پ��������	�����	�����ھ׾־Ҿپ�������	�
���	�������������������������*�6�9�5�*��������ŔŇ�n�e�U�b�{ŇŔŠŹ������������ŭŠŔ�T�M�D�E�H�[�z���������������������z�a�T�ɺú������������������ɺֺ̺���ܺ˺ɻ-�(�+�-�0�:�F�S�T�^�S�P�F�:�-�-�-�-�-�-�F�C�:�9�;�K�S�_�l�x�����������}�l�_�S�F����������������������������������������x�v�}���������������������������������������s�u²������#�<�a�a�U�'����$���3�L�Y�e�r�~�����������~�e�Y�L�3�$���������ùϹܹڹϹù��������������������T�T�M�M�T�X�a�m�z�����������z�z�m�a�T�T�����������������ùܹ����չù����������������������������ĽɽнӽнĽ���������������ݽٽ�����(�6�;�4�,�(��D|D�D�D�D�D�D�D�D{DvDuD|D|D|D|D|D|D|D|D|�`�\�V�S�I�G�F�G�S�`�l�s�y�����y�r�l�c�`����ĿĽĻļĿĿ������������������������ } H B q > J 1 ] @ p l s P , J = 2 f Q y V Z 6 M =   ? 9 2 ' ( ' J * T F { Y ? y A % ? i u ; : 6 ( u W G ' ! K R [ + 0 >    �  j  �  �  �  �  B  O    �  �  )  a  �  �  �  +  ]  v  2    �    �  l  �  �  �    �  �  j  �  �  �  .  �  z  �  �  K  �  Q  �  !  �  �  I  [  �  �  \  O  l  �  d  k    L<e`B���㻃o;�o�D�����
�ě��D����-�o�m�h�D���aG�������ͼ�/�0 ŽY����
���
���
�\�ě���t��@��@��8Q�0 Ž49X�e`B�L�ͽy�#����T����;d�T���8Q콁%�T���}󶽁%�}󶽝�-��%���+������罅���Q�����`��j��"ѽ��m���ͽ����G���l����#BpDA���B�2B[B�B�B��B�B�eB� BHA���B!�B4�B��B>mB�B!�BEB��B	��B��B�6Bg�Bn�B�TB(��BpMB��A���A��5B�MB*�*B��Bc�B
[�B��BpB#B(B	f[B1�B�B
|�B�B"��B'+B)�B8*B�3B
�BpBt�B��B3�B&b�B&v B8�BbB��BP�A���B�uB�YBB��B,B:�B nB/2B�?A���B?�B%�B��B@�B��B! BM�B�B	GpB�	B�@B<cBOWB��B(��B?uBR�A���A�~B��B+6�B��B��B
��B��BG�B4�B(��B	@8B�B��B
hB��B"��B'4�B)�B>QB@-B	�DBD�BI�B��B >B&AzB&BB@BC�B�UA�.�A��nA��A�+@@�|A��pA��KA�i�A�O�A�I�@��cA�[fA�Z�A��8Aͮ�A�0�A��>@�~�A��BAvSA}��@���@�e�A}��A�DAٽ�@�U�AT��A��|A�@�B�B��As��A�7�A�3�A���A��5B�?Aʌ�A�	AX��AV�LA���A�$DA��T@2��@�@��xB�{A���A�ܓ?�p�>}��A��F>�7A#�$A38�C���A��A��A��A�AZA��:A��@DbA��A�-LA�l�A��fA�{C@�A���A��A��A͂EA�r&A��j@���A��3At�?A|��@���@��A}��A�	ZA�7Q@��
AS��A��#A�}�B��B	�Ao?sA��A�yA̓oA�'~B�A�z�A�~CAY AW/A�7?A��<A�}�@3�
@| k@��B��A�sA���?��K>J��A�v�>Fm�A!��A5QC��A:KA��<      $                     V      4      /               &            K   L   ,                                 C   
         	                         	         !   C            #                                          9      A      %               '            3   '                              '      5               #         #      '                  =   !         !                                          %      '                                 /   #                              '      %                        #      '                  ?   !                        N�wCOoޙOrN%��N��N�d�N�2	N0�mO�0�N���O� Nk<�O+�O�N*ŤN`�lN��N�8N"��N��M��PNIVO�pO�O2{O+ mO0߃N�8�Nn�{N�tjN�^=O�x�O���N�bO�o�N��M˼�O���Nqk{O]�N���O2�O�y�O���O�N�jN��N� �N9��O_�4P�\O�`sNN��OS�OM��N���N�3eNbn�N�C}O�\  Z  �      l  �    D    �  1  _  A  �    �  ;  �  \    p  '  1    '    �  �  �  �  �  :  �  �  Z  �  �  �  p  �  �  �  �  Q    L  P  �  K  �  3  �  �  �  '  �  +  '  `  t<��
;�`B<t�<o;D��:�o:�o:�o�+�o���t���`B�#�
�e`B��o��`B�C���C���t���t���j�+�,1�+�����o�C���w�#�
�#�
�#�
�'u�0 Ž0 Ž49X�49X�8Q�P�`�D���P�`�P�`�P�`�Y��ixս�+�m�h�m�h������������-�\��Q콼j��^5���`��G���������������������amz����������znba\\a`anz�����������znca`"#,/<HMHF</#"""""""")16A66)(����������������������������������������9<HLMJH<9:9999999999�����
##
��������pz����������zwppppppO[t������������h[VNOST\alhedaTRQSSSSSSSS��������������������()5BCKNOQNB5)#./00/)#��������������������#/5;<=<6/# ����������������������������������������Q[]cghjg[ZYTQQQQQQQQdgttt{���tqgdddddddd�����������������{����*,21)�����������������������������������������������	#/4:<=<6/# 	bhn{���������{xnba]b� ���������05BCKMDBA5/+00000000jmqz��������zmihjjjjQTUamtwxxmmmaVTQQQQQ��������������������������������������������������������������)59;9??;)���it�����������wtmiiii))/.6897765)))))))))������������������������������������IPU_bn{�������{nUICINNT[gqt�����tg[TNNN��������������������)5[fi]Z[]TB5'$$tz}�����������xtmpst�������
"�����������������������������AIU`bfglbUKIF>AAAAAAz{~��������������{z��������������������M[htsrpormhf`[VOGDDMBENg�������������[NB��������������||���[[hhpnh[PQ[[[[[[[[[[nnz���������zunnjknnFHKUanvz}~znaUOHEEF/009<CILLKII?<<:50///038<?IIMPSSPI<60///<?<;/%##/<<<<<<<<<����
����������������������������������������������������������������������������������,�6�8�6�*���������s�r�g�b�h�s�|�����������������������|�s����������������������������������������ɺȺźɺϺֺ���������ֺɺɺɺɺɺ�����������������������������������������¦¥¦²¸¾²­¦¦¦¦¦¦¦¦������(�.�1�(������������s�i�Z�F�C�F�Z�s�����������������������s���������������������������������������һx�m�u�x�s�j�c�l�x���������������������x�g�e�Z�U�Z�g�s���������s�g�g�g�g�g�g�g�g�5�(����������(�5�?�A�M�N�O�N�A�5�5�4�(�(�#�'�(�5�@�A�N�P�Y�X�U�R�N�A�5�5ùóíìììùú��������ùùùùùùùù�����������������������������������������H�F�<�2�3�<�H�K�U�X�a�i�f�a�U�J�H�H�H�H�f�Y�M�@�4�2�4�9�@�D�M�Y�f�h�r�s�s�r�g�f�������)�.�)�#�����������������������ĿʿƿĿ��������������������ݿտѿѿѿݿ��������ݿݿݿݿݿݿݿݻ��x�_�F�6�2�7�F�����ܻ������лû�����������'�4�M�Y�k�����u�f�M�'���Ŀ¿¿Ŀȿѿݿ߿�������������ݿѿ��������������������#�*�(�#���
�������O�B�6�-�/�2�6�B�H�O�[�h�r�r�t�x�t�h�[�O�����������������ʼϼּؼټؼּм˼ʼ����ʾľȾʾ׾������ �����׾ʾʾʾʾʾ��a�]�`�a�m�z���������z�m�a�a�a�a�a�a�a�a�)�!�����)�-�6�;�B�J�L�J�B�6�)�)�)�)��ƴƳƭƮƳ���������������������������������������������$�0�9�7�1�$��������m�V�P�T�`�y���������Ŀ޿������Ŀ����mŹŷůŭŬŬŭŹ��������������żŹŹŹŹìÓ�}�n�\�R�S�aÇàù��������������ùìùôóìççìòù��������������ùùùù��������þ�������������������������������{�r�b�]�T�M�I�>�I�V�o�{ǍǖǖǔǒǑǈ�{ÓÇÇ�~ÆÇÓÚàäèàÓÓÓÓÓÓÓÓ���s�i�Z�W�M�K�N�W�g�s���������������������������������	�����	��������ھ׾־Ҿپ�������	�
���	�������������������������*�6�9�5�*��������ŔŇ�n�e�U�b�{ŇŔŠŹ������������ŭŠŔ�T�M�D�E�H�[�z���������������������z�a�T�ɺ��������������ºɺʺֺ����غֺɺɻ-�(�+�-�0�:�F�S�T�^�S�P�F�:�-�-�-�-�-�-�_�^�S�F�F�D�F�S�W�_�l�x�������{�x�l�`�_����������������������������������������x�v�}����������������������������������������t²¿������#�a�`�T�&����$���3�L�Y�e�r�~�����������~�e�Y�L�3�$���������ùϹܹڹϹù��������������������T�T�M�M�T�X�a�m�z�����������z�z�m�a�T�T�������������������ùϹܹ���ܹѹù������������������������ĽɽнӽнĽ����������������������(�5�:�4�*�(��D|D�D�D�D�D�D�D�D{DvDuD|D|D|D|D|D|D|D|D|�`�X�S�J�G�G�G�S�`�k�l�m�y�����y�p�l�`�`����ĿĽĻļĿĿ������������������������ z 6 D q > J 1 ] 6 p y s J - G 3 ' ) Q y V Z 5 > ;   ? 9 - " ( ' J * O F { Y ? N A % ? i u 3 : / ( u Y G ' ! ! R N + / >  �  �    �  �  �  �  B  F    �  �  �  B  L  h  �  	  ]  v  2  �  '  7  �  l  �  �  �  �  �  �  j  �  T  �  .  �  z  �  3  K  �  Q  �  �  �  �  I  [  s  �  \  O  �  �  $  k  �  L  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  Ag  O  S  W  X  Q  I  =  .      �  �  �  �  �  �  �  �  �  �  ]  o  |  �  �  �  �  o  U  2    �  �  :  �  e  �  q  �  �  �     
        �  �  �  �  �  �  h  D    �  �  y  0   �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  `  W  Q  K  :  $    �  �  �  �  �  h  K  -    �  �  u  �  �  �  �  �  �  �  �  y  p  f  Z  O  A  .    	  �  �  �        �  �  �  �  �  �  �  |  g  M  3    �  �  ~  &  �  D  C  B  B  A  ?  <  8  5  1  $    �  �  �  �  �  �  z  g  �    L  s  �  �        �  �  �  �  o    �  6  �  �  �  �  v  l  b  W  L  @  4  '              �  �  �  x  H  �  �  �            ,  0    �  �  C  �  �    }  >    _  \  Y  V  S  P  M  I  D  ?  :  5  0  )  !        �  �  t  �  �  �  �  �    1  @  9  #  �  �  �    y  �  �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  i  U  ?  '  �  �  �  �  �  �      $  !    �  �  �  q  G    �  �  �  X  "  �  �  �  �  �  �  �  �  �  �  N    �  �  :  �  �    �  Q  B  r  �  �  �  �    -  :  :  /    �  �  �  \    �  S  �  *      b  �  �  �  �  �  �  �  �  �  d  9    �  �  P  /  \  W  S  N  J  E  A  ;  4  -  &          !  '  -  2  8       �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  m  _  R  p  i  a  Z  S  K  D  =  5  .  1  >  J  W  d  q  }  �  �  �  !  %      �  �  �  �  �  �  �  �  �  <  �  ^  �  /  �   �  �  '  .  1  -  #  	  �  �  `    �  �  M  �  �    I  
  S  Z  �  �            �  �  �  m  ,  �  p  �  �    �  j  #  %  &  "    
  �  �  �  �  �  a  C  )    �  �  �  �  �    �  �  �  ]  <    �  �  �  p  >  
  �  �  ~  J    �  d  �  s  f  W  H  6  $    �  �  �  �  �  u  R  -  �  �  L   �  �  �  �  �  �  �  �  �  �  �  �  s  Y  4    �  �  �  Q    �  �  �  �  �  �    k  S  :       �  �  �  �  �  �  q  ^  �  �  �  �  m  U  <  $  
  �  �  �  u  =  �  �  E  �  �    �  �  �  �  �  �  �  w  e  Q  4    �  �  r  8  �  �  w  3  :  6  (      �  �  �  �  b  /  �  �  �  h  @    �  �  #  �  �  �  u  Z  6    �  �  q  3  �  �  d  /  )  >      +  �  �  �  �  �  �  t  f  X  F  .      �  �  v  %  �  _   �  �  .  B  L  P  Y  C    �  b  	  �  �  U    �  K  �  4  1  �  �  �  �  �  �  �  �  �  �  �  n  U  1  �  �  o  %  �  �  �  �  y  o  d  Y  O  D  9  .  &                �  �  �  �  �  �  u  X  6  %    �  �  �  �  ^  -  �  �  i    �  �  p  d  X  K  <  ,    	  �  �  �  �  �  }  b  <    �  �  L  �  �  �  �  �  �  �  �  �  o  P  *  �  �  �  P    �  �  5  �  �  �  �  �  �  �  �  �  �  �  s  O  #  �  �    �   �   L  �  �  �  �  �  �  |  f  L  0    �  �  �  �  I  	  �  q    �  �  �  �  �  �  �  �  �  }  [  3    �  �  �  {  F  �  	  Q  E  8  &      �  �  �  �  �  �  �  �  s  Y  D  E  `  �    �  �  �  �  �  x  B    �  �  �  �  a  A    	  �  Q  �  	  6  G  K  J  D  ;  -    	  �  �  �  �  l  @    �  �  �  P  N  M  D  8  +      	       �  �  �  �  h  )  �  ;   �  q  �  �  �  �  �  �  �  �  �  |  P    �  }    �  �  :   ~  K  +    �  �  �  �  e  G  !  �  �  �  {  O     �  �  x  9  �  �  {  ^  5        �  �  �  �  �  d    �  J  �  D  �  1  %    �  �  �  �  g  (  �  �  �  �  g     i  �  �  &  a  �  �  q  R  /  �  �  �  ;  �  �  <  �  i    �  �  Q  !  �  �  �  �  �  �  �  �  �  �  �  �  x  d  L  /    �  �  �  �  �  �  k  O  1    �  �  �  _     �  �  I  �  �  a    y   �  �  �  �  �    �  �  �  t  >  �  �  p  "  �  �  2  �  m  �  �  �  �  �  �  �  �  j  S  <  "    �  �  �  �  f  E  !  �  �    %    �  �  �  �  Y  -  �  �  �  K    �  |  (  �  f  '  �  �  �  w  N    �  �  n    �  1  �  +  �  �  ?  �   �  T  [  _  ]  Y  S  G  7  $      �  �  �  �  �  �  �  �  4  t  a  M  9  $    �  �  �  �  �  b  G  ,    �    q  �  �