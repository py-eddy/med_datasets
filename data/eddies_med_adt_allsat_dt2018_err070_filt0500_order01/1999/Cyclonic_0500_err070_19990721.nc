CDF       
      obs    U   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�������     T  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P��     T      effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       =Y�     T   T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @F��Q�     H  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q�    max       @vd(�\     H  .�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @O�           �  <8   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ɯ        max       @�T          T  <�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �	7L   max       =0 �     T  >8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B2�     T  ?�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��]   max       B0��     T  @�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =���   max       C��)     T  B4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =���   max       C��'     T  C�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          e     T  D�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G     T  F0   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7     T  G�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P���     T  H�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�5?|�i   max       ?�M����     T  J,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =Y�     T  K�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @F��G�{     H  L�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ᙙ���    max       @vd(�\     H  Z   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P            �  gd   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ɯ        max       @��         T  h   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�     T  id   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?q4�J�   max       ?�M����     �  j�   
            1                     &               
      d   )                  '               9         "               	                                 	   \            =      	            	   
                              	   A      	                   
            NC�N�W�O<��N+�P#N�[ OgYN��|OQ��NA��P?hO+LNT\�O�(OK�O��N��zN�(�P��PA)�N@��N��dOj�O8cN�P&��N)<,O��O'�N�z�P	��O��OO�wO�u�OA�;N(�6NF��N2*N��O��iO�x	N�7�O��N�h4N2� O[[Og��N���O�aN���P6��N.��NO��M���O�N�O�kN�P�Op*\O.]~N�V]N�t?Nk�?N��%NTgNԲ�O	@O6lSO(�>O-'�O�:O� bNB�5P?#O�N���O��OQ�AOE�O	��M�>�N�R�N� RN�lO.�!O�d�=Y�<���;ě�;��
;�o;D��;o:�o�o�o���
�ě���`B�o�t��#�
�D���D���T���T���T���T���u�u��o��o��C���t���t����
���
��1��9X��9X��9X��9X�ě��ě�������/��/��`B��h���o�+�C��\)��P�'0 Ž49X�49X�<j�D���D���P�`�]/�]/�aG��aG��e`B�ixսixսixսm�h�m�h�q���q���q���q���u��%��o��o��7L��7L��7L��7L���P���w�������j��j��������������������zz{�����������}zzzzz��������������������

^gz�����������ma[WV^�����������������������
#/<HH;#�������)6>BFB@6+) #/<HU[\UH:5/.))('# ��������������������Hal����������znUJ@@H��������������������55BDFFB<5++355555555)6BO_a]PLB6.*)#����
 ����������JO[ht������th[[ONKJJKO[hjtvwtmh[WOKFKKKK���������������������	5B^����bNB5%�������������������������������������������

����������)+/.)������IO[ht�������th[WPOI�������������������#0;Oh��������uC6*��

����������>BOThoolihe[YOKHB@8>��������������������#/<HNIHA<//#������������������|��������������������������������������������������������������������������������������������������		������������������������������dgot����������|tsgdd������� ������ 5BNgt�����tg[NB6(& 3<=AIUZ^bcb^UIB<:233+/;BGGHIHHG;5/%%'*++?BCOQ[dhkh`[OB??????����������������������������������������RU[amz������zma[TSQR���� ��������������������������������������������������<?Uanz���������nUH<<SUafkjaUTRSSSSSSSSSS���������~���������<<INOIIA<;<<<<<<<<<<�������������������������
#1/,#
���������������������������������������������������������������������������������������������)46776)$egstv�����tg][eeeeee����������������������������������������5;<HTXaab^XTIHF;7335��������������������t����������������yvt���� ������������������������gn{����������{uqqigg����������������������5[`W?0��������� )6=A6)�����IOW[hrqnha[ZOCIIIIIIrt������������tnhlrr
!#%(++)%#
���"#)/4<>ABA<:/#"#""#"#)02<>GJIIA<0(#����������~��������CHRU^acdaaWUOH@=CCCCknvz������zonikkkkkk������������dgt�����������trgdad���������������������U�Q�T�U�Z�a�f�i�e�a�U�U�U�U�U�U�U�U�U�U��~�r�g�r�r�|��������������������������������������$�0�4�=�=�8�0�$������������������������������������������0�#����	��0�<�b�n�{ŅŇŃ�t�i�b�U�I�0����������������������������������������àØàãáâáìíù��������������ùìà�����������������������������������������
������	��#�B�C�H�U�W�U�H�<�/���
�G�G�;�.�"��"�+�.�3�;�G�T�]�T�I�G�G�G�G�5��	�������������"�H�m�z�����{�m�T�H�5ÇÃ�z�v�r�w�zÇÊÓÝàìóïìåàÓÇ�a�`�Y�a�m�z���������z�m�a�a�a�a�a�a�a�a������������.�;�B�H�;�.�"������лϻͻ̻˻лܻ�������������ܻкY�S�S�Z�g�n�r�u�~���������������~�r�e�Y�����������¿ĿѿԿݿ��ݿֿѿĿ������������������Ǻɺֺۺ����ֺܺɺ������������Z�L�O�g�����������/�J�J�/��������������s�g�b�b�s�������������	�
�����������ʼ��������ʼʼּټּּܼʼʼʼʼʼʼʼʼ��������������������������ļ�����������¦¦¯±²¿����������������¿²¦���x�u�t�o�i�l�t�x����������������������ɺȺ��������ɺѺֺߺֺкɺɺɺɺɺɺɺɿ	�׾��������������ʾܾ�����'�,�+�"�	�0�*�0�1�=�I�V�_�V�I�D�=�0�0�0�0�0�0�0�0�������������������������������������������	�����������	���"�*�.�.�/�/�"��6�2�2�2�2�6�B�C�O�V�O�O�J�B�6�6�6�6�6�6������8�M�f�l�n�w�{�����|�l�Y�M�4�������ɾݾ���"�&�-�2�2�(�	�����׾ʾ�������������������������������	��
�����ֺ������������ɺֺ���-�:�E�8�!����������~�w�t�v���������������������������
������������
�������
�
�
�
�
�
���	����	���"�'�$�"���������������#��������������C�<�6�1�.�*�(�*�6�C�J�O�[�\�h�i�h�Z�O�C������׾����������׾�	��"�.�8�?�.����о˾ܾ�����	��� �����	�������������������������������������������������������������������������������������ϹŹù����������ùƹϹӹ׹ԹϹϹϹϹϹϿ�������������������������������������y�u�m�h�j�m�y������������������������ƳƪơƠƤƧƳ����������������������EiEeE`EiEuE�E�E�E�E�E�E�E�EuEiEiEiEiEiEi�@�3�'����'�/�3�@�N�Y�e�r�����r�e�L�@���������������������������������������޹�����q�j�i�x�����Ϲ��5�6�'�����ù������(�5�:�8�5�(�����������ĿĿĿ̿ѿݿ�����ݿѿĿĿĿĿĿĿĿĻ-�,�(�-�:�F�H�F�D�:�-�-�-�-�-�-�-�-�-�-�x�n�j�m�t���������ĿǿĿĿ¿����������x���ݿѿͿϿѿؿݿ߿�������	������ŠşŔœœŔśŠţŭŹŻ��ŹŭţŠŠŠŠ�O�B�<�;�B�F�B�:�B�O�[�l�tăĀ�{�t�h�[�O������ŽŹŹ��������������������������߿ݿܿѿĿ����������Ŀ̿ѿݿ�����ݿ���������� �����"����������������ìåæìñù��������þùìììììììì�=�;�0�0�0�3�=�I�U�V�]�\�V�I�=�=�=�=�=�=������������
���������������������������������������������������������������(�5�:�A�B�A�;�5�(��������������!�.�9�G�H�H�M�G�?�.����ŔŇ�}�{�n�n�n�{ŇŔŭŮŹ��žŹŷŭŠŔ�ܻлŻλܻ��������%�%��������ܼ4�.�)�(�4�?�@�M�Y�f�g�p�o�f�e�Y�M�@�4�4��޻޻׻ܻ����4�M�Y�P�M�A�4���������������
���"��
��������������������÷ëäß×Çàù������1�;�9�/������÷���i�^�f�y�������Ľнѽڽ���ٽнĽ����Ľ����������Ľн۽ݽݽݽӽнĽĽĽĽĽ��/�-�$�(�/�8�<�H�T�U�a�h�a�[�W�U�H�<�/�/D�D�D�D�D�D�EEEE*E7ECEFEOEFE*EEED�D�D�D{DxD{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������������������������ĽĽȽĽ½�������²±¬²½¿����������¿²²²²²²²²�����������	���"�#�"���	���������������������������ʾоʾþ��������������ʼʼԼ������������������ּ�čĉăčęĚĦĳĿ������������ĿĳĦĚč������ĿļĿ���������
���#����������� ~ 4 2 X ( ` j 6 9 k j  T C L g 4 - 4 < ; T O _ R U l L { T D x P r S g O C E w A 7 X W ` B D \ , A Z / L ) ' @ / 6 > ^ R V A G : % y < R   ] . _ > B   V c  n * 4 v P >    �  �  �  @  �  �  c  �  �  ~  z  g  l  *  W  �  �  �  �  \  c  �  �  �  H  l  p  ]  �  �  �  �  f  �  �  u  S  W    1  �  �  F  �  D  (  �  �  f  �  �  F  z    �  o  �    �  
  �  �  �  9    N  1  �  t  R  k  [  �  (  �  -  �  \  -    �    k  �  8=0 �<�t���o:�o�8Q�%   ��C����
�u��o���49X�#�
���㼴9X��󶼬1���
��l��]/��t����
���\)��1�aG���1���\)�������-�''aG��C�����h��`B�\)�]/�]/�o�'49X�\)�@���%�Y��y�#�L�;	7L�<j�@��Y���/��t��q�������u��%�����C��q���}󶽝�-���w��7L��\)��O߽�����O߾o�\��t��� Žȴ9��vɽ�9X���㽴9X�� Ž�������;dB��B��B�xB$~�B .vB gB#�B��B�B!ĉBFB��Bg�B��B#�B�[Bb`B �B�BNgB ��B#�{BqB$�B#KB2�Bw�B({B!~�BsHB @�B�sB�uBp�B@�B L)A���Be$B	��BBmB�sB&�A��B��B)�$B*��A��aB�qB"p�B��B�jB'BضB&�-B�CB��B~�B��B1TB �B�BB	��BPB{�A��tBB
�BڰB�`B)[�B��Bq�B�B`B
e�B7AB1^B%��B
v�B��BL�B.CB	�VBIBB42B�TBeiB$L@B >�B �B��B�B�,B!�'BL=B��BM�B@@B#BBNBU[B °BA4B�B ��B#ܐB(}B3nB#C�B0��B��B=�B!uB��B��B�*B��B-#B��B @�A��]B�B	�7B@�B	;B&pmA��ZB?pB)��B*��A�}	B��B"B�B�iB@5B�BʷB&��B�GB��B�,B?�BC>B=�B?�B��B	�BR�B�!A��BJ~B
�sB)�B�B)92B�$BFSB:8By�B
�B@-B@oB%�:B
A�B�XB?�B.?�B	��B��A��d@�B	K"AJ�A�}�AK��A�/�A�q�A��pAb��A��A�}}A���A[�d@� �?��SAy�@6a~A�� A��)@��a@�y�A��@���@3��AV��B
�A��aA�e�A���@�=yAW��A�{�@B�EA���A�/A�ZA�kbB�ATt�AYBFA�f[A���>`��A�K�Ao��B2KC��)?�T�A�U�=���A�~4A|�>@z��Ar{�A~�!A��/Aګ�A�U�Az��B̏A�\mB7�A��iA��A�ùA\A��@�ED@���@Ę�A��\A�	BA"R�A'|�A�%C�r�C��nA">XA���AZ��AM��AVA�k�A�,�AƁ�@��wB	��AI)�A�~lAL ^A��A��A�jAbO�A�~�AʀA���A\�I@�;�?���Ay��@3v6A�{�A� �@�h@�MsA��v@�1�@4d�AU	HBtvA�KA�~�A؏�@��
AZ��A�e�@8u�A���A�waA��/A��$BwAS'�AY?�A�f&A���>?��A�QAo BB>�C��'?Ѕ�A��=���A��A{ �@{�-Ar�A�A�1Aڇ@A���Ay��B	 �A̒$B>�A��A���A�؁A�A�|�@��@��@��A� yA�~'A"��A'��A�GC��pC��A"��A�z�AY`@AM Az5Aߗ�A��      	         2                     '               
   	   e   *                  (               :         "      	         
                                 
   \            =      	            	   
                  	            
   B       	                                              #                  1                        G   1                  -               +   #      -                  '                           !      5                                                                  1   '                                                                  /                        7   -                  +                  #      -                                                   !                                                                  1                                    NC�N�W�O0@�N+�O��N�[ O�NN��|OQ��NA��P�N��NT\�O�(OK�O^�N��zN�(�P���P#�N@��N��dOj�O0�N�P��N)<,N��N�a�N�z�O��6O��ON��O�u�OA�;N(�6NF��N2*NI)�N�H�O:i�N�7�N�[�ND��N2� N�3�ON�FN���N�v�Nu��O�c(N.��NO��M���O���ON?N�P�OZ�O.]~N�V]Ne�aNk�?N��%NTgNԲ�O	@O$K�O(�>O-'�O�:O� bNB�5P8�KOz�N�]�NљOQ�AOE�N�'M�>�N�R�N� RNt�N�:O�d�  �  �  N  �  �  �  �  d     '  #  	  e  �  �  �    �  	!  �  �  �  p  �  �  4    1     �  �  �  E  �  �  C  �  a  3  �  �  �  	  r  �    f  �  X  �  ,  �  �  j  	�  !  �  �  �  o    �  O  �  �  �  Y  :  �  !  4  �  	�  �  �  :  	�  �    �  0  �  �  
  g=Y�<���;��
;��
��C�;D���o:�o�o�o��`B�T����`B�o�t��49X�D���D����㼓t��T���T���u��t���o��t���C����
��1���
�#�
��1��/��9X��9X��9X�ě��ě���h�0 Žt���`B���o�o�C��t��t��D���0 Ž�7L�49X�49X�<j�P�`�H�9�P�`�aG��]/�aG��e`B�e`B�ixսixսixսm�h�q���q���q���q���q���u��o���P�����\)��7L��7L��O߽��P���w������9X�����j��������������������zz{�����������}zzzzz��������������������

kmz�����������zmhbck�����������������������
#/<@<6/&#
�����)6>BFB@6+) #/<HU[\UH:5/.))('# ��������������������Han�����������znVFBH��������������������55BDFFB<5++355555555)6BO_a]PLB6.*)#����
 ����������KO[dhrt����th[ONLJKKKO[hjtvwtmh[WOKFKKKK���������������������)5BNpxw]VPB5.���������������������������������������������

����������)+/.)������OY[ht��������|th[SOO�������������������%2>Oh�������uC6*$ %��

����������FOW[hlmkhhc\[ONKEBFF��������������������#/<HNIHA<//#���������������������������������������������������������������������������������������������������������������������		������������������������������fght����xtpgffffffff��������������>BNTgt{���utg[NLB;8>3<=AIUZ^bcb^UIB<:233,/;@FFHHHGC;7/('(+,,BBEOT[ahih^[OBBBBBBB����������������������������������������STV\amz�����yma]TRS���� ����������������������������������������������������FUanz������znaUHBBFSUafkjaUTRSSSSSSSSSS���������~���������<<INOIIA<;<<<<<<<<<<����������������������
#-*#
������������������������������������������������������������������������������������������������)46776)$egstv�����tg][eeeeee����������������������������������������5;<HTXaab^XTIHF;7335��������������������t����������������yvt���� ������������������������gn{����������{uqqigg����������������������5B[^V>/�������&)(�������LOX[hppmh[OKLLLLLLLLotv����������ytkoooo
!#%(++)%#
���"#)/4<>ABA<:/#"#""#"#0<=EGE<30.#����������~��������CHRU^acdaaWUOH@=CCCCknvz������zonikkkkkk��	���������fgt}��������utkgfcff���������������������U�Q�T�U�Z�a�f�i�e�a�U�U�U�U�U�U�U�U�U�U��~�r�g�r�r�|�������������������������������������$�0�3�<�<�7�0�$������������������������������������������0�-�#� ��#�+�<�I�U�b�n�p�u�o�g�b�U�I�0����������������������������������������ìãåçæççèìðùþ������������ùì�����������������������������������������
������	��#�B�C�H�U�W�U�H�<�/���
�G�G�;�.�"��"�+�.�3�;�G�T�]�T�I�G�G�G�G�9�"�������������"�/�H�a�m�����z�a�H�9ÓÊÇ�z�z�w�z�}ÇÓàèìðììàÝÓÓ�a�`�Y�a�m�z���������z�m�a�a�a�a�a�a�a�a������������.�;�B�H�;�.�"������лϻͻ̻˻лܻ�������������ܻкY�T�T�Y�[�e�h�o�r�~����������~�r�e�Y�Y�����������¿ĿѿԿݿ��ݿֿѿĿ������������������Ǻɺֺۺ����ֺܺɺ����������r�e�c�g�r�����������	�,�,�"�	�������������s�f�k�����������������������������ʼ��������ʼʼּټּּܼʼʼʼʼʼʼʼʼ��������������������������ļ�����������¦¦¯±²¿����������������¿²¦���x�x�v�q�k�l�r�x�����������������������ɺȺ��������ɺѺֺߺֺкɺɺɺɺɺɺɺɾ׾��������������ʾ׾߾��$�)�'�"��	���0�*�0�1�=�I�V�_�V�I�D�=�0�0�0�0�0�0�0�0�����������������������������������������"����	�� �����	����"�'�+�*�$�"�"�6�2�2�2�2�6�B�C�O�V�O�O�J�B�6�6�6�6�6�6������<�M�Y�f�h�p�s�n�a�Y�M�@�4�'������ɾݾ���"�&�-�2�2�(�	�����׾ʾ��������������������������������������ֺ������������ɺֺ���-�:�E�8�!����������~�w�t�v���������������������������
������������
�������
�
�
�
�
�
���	����	���"�'�$�"���������������#��������������O�E�C�9�9�C�O�P�\�a�\�R�O�O�O�O�O�O�O�O�׾Ͼʾþʾ׾��������׾׾׾׾׾׾׾׾���������	�
��������	�������������������������������������������������������������������������������������ϹϹù����������ùĹϹѹֹϹϹϹϹϹϹϿ�������������������������������������y�v�m�i�l�m�y��������������������������ƳƬƣƢƧƳ����������������������EiEfEfEiEuE�E�E�E�E�E�E�E�EuEiEiEiEiEiEi�@�?�3�3�,�3�@�L�Y�e�r�p�e�d�Y�L�@�@�@�@���������������������������������������޹����}�y��������������� �����չù��������(�5�:�8�5�(�����������ĿĿĿ̿ѿݿ�����ݿѿĿĿĿĿĿĿĿĻ-�,�(�-�:�F�H�F�D�:�-�-�-�-�-�-�-�-�-�-���y�p�l�n�u�����������Ŀ¿ÿ������������ѿοϿѿԿڿݿ�������������ݿѿ�ŠşŔœœŔśŠţŭŹŻ��ŹŭţŠŠŠŠ�O�G�B�<�<�;�B�I�O�[�h�k�tĂĀ�z�t�h�[�O������ŽŹŹ��������������������������߿ݿܿѿĿ����������Ŀ̿ѿݿ�����ݿ���������������!����������������ìåæìñù��������þùìììììììì�=�;�0�0�0�3�=�I�U�V�]�\�V�I�=�=�=�=�=�=������������
���������������������������������������������������������������(�5�:�A�B�A�;�5�(��������������!�.�:�G�L�G�>�:�.�!����ŔŇ�}�{�n�n�n�{ŇŔŭŮŹ��žŹŷŭŠŔ�ܻлŻλܻ��������%�%��������ܼ4�.�)�(�4�?�@�M�Y�f�g�p�o�f�e�Y�M�@�4�4��޻޻׻ܻ����4�M�Y�P�M�A�4���������������
���"��
��������������������øìåàØÑÑàù������0�:�8�-����ø���n�s�y�����������ĽҽڽսнĽ����������Ľý��������ĽнڽܽѽнĽĽĽĽĽĽĽ��<�1�/�(�+�/�<�C�H�L�U�_�W�U�R�H�<�<�<�<D�D�D�D�D�D�EEEE*E7ECEFEOEFE*EEED�D�D�D{DxD{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��������������������ĽŽĽ���������������²±¬²½¿����������¿²²²²²²²²�����������	���"�#�"���	���������������������������ʾоʾþ�������������������������������������čċąčĐĚĤĦĳĽĿ��ĿľĳĭĦĚčč������ĿļĿ���������
���#����������� ~ 4 2 X  ` s 6 9 k j  T C L V 4 - + = ; T O S R X l = k T 8 x \ r S g O C 6 ] ? 7 W Y ` = E Z & < 0 / L )  4 /  > ^ Y V A G : % � < R   ] . ^ > 1   V c  n * 4 8 B >    �  �  }  @    �  �  �  �  ~  �    l  *  W  Q  �  �  G  �  c  �  �  f  H  g  p    T  �  ;  �  �  �  �  u  S  W  \  �  �  �  $  �  D    �  �    �  �  F  z    k  I  �  �  �  
  �  �  �  9    N  �  �  t  R  k  [  �  �  �  �  �  \  �    �    �    8  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  �  �  �            �  �  �  �    ^  <    !  1  T  x  �  �  �  �  �  �  �  �  �  �  �  �  |  q  f  \  S  K  G  B  L  M  L  I  E  ?  8  -    �  �  �  g  )  �  �  Q    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Q  v  �  �  �  �  �  �  �  �  �  X  �  �    �  �  X    -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  n  c  W  L  �  �  �  �  �  �  �  �  �  �  G  �  �  �  �  o  3  �  �  h  d  c  b  a  ]  X  S  M  H  B  ;  4  -  $      �  �  �  �                     �  �  �  �  �  �  s  X  7    �  '         
     �   �   �   �   �   �   �   �   �   �   �   �   �   �       "            �  �  �  �  �  �  �  �  O    �  �  �  �  �  	  	  �  �  �  �  F  �  �  ;  �  J  �  �  �  �  -  e  _  Y  T  N  I  C  C  F  I  M  P  S  X  _  g  o  v  ~  �  �  �  �  �  o  Y  >  #    �  �  �  �  z  ^  >     �   �   �  �  �  �  �  �  �  �  �  �  �  �  t  G    �  �  b  &    %  Z  �  �  �  s  c  P  7    �  �  �  i  &  �  ~  "  �  �  �        �  �  �  �  �  �  �  �  �  n  Y  D  0      �  �  �  �  �    y  s  i  `  W  O  L  N  Q  U  X  W  V  R  L  E  �  �  �  	  	  	   	  �  �  ]    �  5  �  0  �  �  2  O  1  a  |    {  n  [  >    �  �  �  �  _  $  �  �  �  4  �  *  �  �  �  �  �  �  �  �  �  x  j  [  H  3      �  �  x  F  �  �  �  �  �  �  �  r  e  U  E  4      �  �  �  �  �  �  p  a  O  9  !    �  �  �  �  �  u  U  :  %  �  �  A  �  �  �  �  �  �  �  �  �  �  �  �  �  `  0  �  �  X    �  [  A  �  z  r  j  b  U  I  <  4  1  .  +  -  1  5  :  @  F  L  R  ,  4  2  +  "  %  &    �  �  �  �  e  G  $  �  �  K  �  7    �  �  �  �  �  �  �  �  �  �  a  A  !    �  �  �  �  �    '  -  1  /  *  #    �  �  �  �  �  c  7    �  �  o  9  �  �            �  �  �  �  �  �  �  Y    �  �  -  �  �  �  �  }  n  ]  M  <  +      �  �  �  �  [  .    �  �  <  �    c  �  �  �  �  �  �  �  l    s  �  �    �  �  �  �  �  �  �  y  f  R  <  &    �  �  �  K    �  �  �  B  �  �  �     (  <  E  1  
  �  �  �  t  B    �  �  �  �  �    �  �  �  �  �  �  h  D  $      �  �  x    �  a  �  =  >  �  �  �  �  �  �  �  �  �  �  �  �  f  E  $    �  �  �  j  C  4  &        �  �  �  �  �  �  �  �  r  C    �  �  e  �  �  �  �  �  �  �  x  i  W  E  3      �  �  �  �  �  h  a  Z  S  M  F  B  E  H  K  M  H  ;  .  !    �  �  �  �  �  !  '  -  .  .  0  2  .  &      �  �  �  �  �  i  R  e  w  �  �  �  �  �       !  (  >  i  �  u  ;  �  �    �    �  X  �  �  �  �  �  �  �  �  �  �  t  P  $  �  �  ]    �  �  �  x  q  i  a  Y  O  F  <  3  )           �           	  	    �  �  �  �  �  �  X  *  �  �  �  L    �  �  �  �  W  b  l  r  q  m  e  V  8    �  �  �    p    �      �   *  �  �  �  �  |  u  m  e  ]  T  K  C  :  0  #    	   �   �   �          
     �  �  �  �  �  �  d  8    �  r     �   T  E  d  ^  N  9    �  �  �  ~  P    �  �  "  �  T  �      �  �  �  �  �  {  _  >    �  �  �  ~  S  !  �  �  \    �  /  3  2  .  +  0  ;  J  U  X  U  N  C  2    �  �  �  W  ~  �  �  �  �  �  �  �  �  �  �  o  S  6    �  �  �  h  2   �  
  
�  
�  
�  
�  ,  $    
�  
|  
  	�  	   �  �  ,  R  1      �  �  �  �  �  �  �  �  �  �  �  �  �  �    v  m  e  \  S  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  K  ,    �  �  �  �  r  R  1    �  �  �  y  R  *     �  	l  	�  	�  	z  	k  	Z  	F  	)  	  �  �  A  �  �  J  �     I  @  ~          	  �  �  �  �  z  T  .    �  �  X    b  �  v  �  �  �  �  �  �  �  �  t  e  V  A  &    �  �  �  T    �  R    �  �  q  U  8    �  �  �  I  �  �  P  %  �  �  v  ,  �  �  �  �  �  h  G  *    �  �  �  �  �  �  �  �  s  V  <  "  o  a  T  G  9  )    	  �  �  �  �  �  �  |  e  J  0     �  �           )  )  )  %  !          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  ]  F  /    �  �  �  p  �  f  O  A  3  '      �  �  �  �  �  �  �  R    �  �  g  $   �  �  �  �  �  �  �  �  �  �  }  u  m  d  \  T  K  C  ;  2  *  �  �  �  �  �  s  c  S  B  /    
  �  �  �  �  �  x  \  @  �  �  t  k  Z  F  /       �  �  �  ^    �  {    �  (  �  4  W  7      �  �  �  c  F  (  �  �  r  1  �  �  �  �  u  :  $    �  �  �  �  �  z  `  E  *    �  �  �  �  c  M  8  �  �  �  �  �  �  �  �  �  �  r  j  g  S  9    �  �  0   �  !       �  �  �  �  �  �  �  �  �  �  �  �  �  j  R  +    4       �  �  �  �  �  �  �  k  ;    �  �  �  b  6     �  �  �  �  �  |  m  ]  I  2      �  �  �  �  {  a  G  0    	�  	�  	�  	O  	  �  y  6  9  "  �  �  #  �  5  �  �    )  �  }  �  �  �  �  �  �  �  �  �  �  j  1  �  �  T    �  H   �  x  }  �  ~  v  k  Y  H  4  !    �  �  �  �  �  �  s  \  E     *  7  :  7  +      �  �  �  �  a  ;    �  �  �  �  �  	�  	n  	3  �  �  r  ,  �  �  C  �  w     �    c  V  z  �   �  �  �  �  }  8  �  �  g  '  �  �  �  D  �  �  J  �  e  �  i  �  �       �  �  �  �  �  �  �  n  @    �  �  Y  �  �  6  �  �  �  �  �  �  y  o  f  ]  V  R  O  K  G  C  ?  <  8  4  0  !    �  �  �  �  �  �  d  G  '    �  �  �  �  �  �  
  �  �  �  �  �  �  �  �  �  �  z  m  `  S  B  +    �  �  �  �  x  e  �  {  f  �  �  K    �  �  a  #  �  �  6  �  �  �  �  �  �    �  �  �  �  �  �  ~  g  ]  J  #  �  �  =  �  �  g  T  =  (    �  �  �  |  a  E  %  �  �  �  >  �  �  T  *