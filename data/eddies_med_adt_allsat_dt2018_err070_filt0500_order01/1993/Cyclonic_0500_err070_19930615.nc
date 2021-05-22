CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?���E��     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P�C�     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�o     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?k��Q�   max       @E�G�z�       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v��\)       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @Q            �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���         4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �,1   max       <t�     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�&�   max       B-�F     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�p�   max       B. �     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�V�   max       C��      4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >~�   max       C��     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       O�Y     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�GE8�4�   max       ?�F�]c�f     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <�o     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?xQ��   max       @E�G�z�       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v�(�\       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q            �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�d�         4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�F�]c�f       cx                     �                  *               
         /   "      #                              %                  	                                    E   
            4         
   	   
         	            S   
               	      N���N��gN�ޜN�[N�0O$�wP�C�Ob��O6שO�N���N�F�P6�QO�ɷOǭN���N��8N��O��N~��P[{P��O�2�O�-lO^m�N�ʓO(Y�N	K�NzOA�"N���O��HO��.P,yvN���O&�7Oj��OH��N�D�NNO��O�+�OT�RO88N��[N6$Or~�N��N��UO	/N�EzP*d�N�ӏO2��O��'N���O�~0O[�O�O�gN�r�N�c_N�hO���N#QCN6�OUT�N��NO��nOo�N���O���OCoN��N�԰O�oN�/l<�o<�o<e`B<o:�o��o��o���
�o�o�o�o�t��#�
�D���D���T���T���e`B�u�u��o��C���C���C���C���C���t����㼛�㼛�㼣�
���
��1��1��1��j��j��j�ě����ͼ�/��/��`B��h��h���o�o�o�t���P��w��w�#�
�#�
�#�
�'0 Ž8Q�8Q�8Q�@��@��D���H�9�L�ͽT���Y��ixսm�h�u��o���㽛�㽧�������������������6<DIU_a]UUIG<7:76666���������������������������������������������������������������������������������$#���������45BDN[gs|���tg[UN.*4��������������������'/5<>GHORSLH<4/*()''aanrvz{|{zna`XY\aaaa��������������������kt�������������{tjgkUan������������nOOUU�����-0,)�����������������������������������������������������������������������%$ ������������������HLalz���������~maTLH(.5B[gorslgaNB0**+'(��������
�����������#%%$
������xz���������zvrssxxxx����������������������	���������������������������������#/<HU_c`UTJ><8/-#46:BNO[ca[VOIB614444-5B[gpt���yg[NB850+-��������������������HRam��������}wmaTKFH���������������|u��������������������TY`mz��������zma]XST�����������������������������������������������������������@COVZ\_ht���thVODA@@it}��������������thi8<>Uan������znaTMH<8��������������������ot|���������tnjmoooost�����trnssssssssss������������������������������������������������������������������
���������HIU`bhfhbUIHGCHHHHHH������������������|�����������������������������������������dgt�������������g``d��������������������05<EIKRU[`_[UI<0-,,0����
#+/(#
�������������������������S[git�������~tg[TPSSQUYanz|�zwnjaa_UNMQQ9<FIOU_YVUJIG<119999���)5BN[gt~���tgNB5)#S[hmkh`[YRSSSSSSSSSS"#04<HE<20,#""""""""r{���������������zr��������

�������#)*(*0/#
��������	����������������������������2Bbn��������tg[N4-2������������������������������������������������������!#+--2;?BC=:/#,/3<FHU]`YUH<71/.-,,àÕàäìù��������ùìàààààààà�ʼ����������ʼּ��������ּʼʼʼ�¦²³¿����¿¶²¦�����������������������������������������H�E�>�G�H�P�U�a�d�k�i�a�\�U�H�H�H�H�H�H�S�M�F�C�F�K�L�S�_�x�������������x�l�_�S�����������4���������������f�4�����ƎƍƄƁ�x�xƂƎƚƣƧƳƺƸƶƳưƧƚƎ�����������������
��$�/�4�4�/�/�#��
���"�����"�#�/�;�H�T�V�X�T�P�H�;�/�"�"�A�A�;�A�N�Y�Z�g�s�}��v�s�g�Z�N�A�A�A�A�!���!�-�1�:�C�F�S�Y�S�S�F�:�-�!�!�!�!�����y�h�b�`�i�y�����Ŀֿ������ݿĿ�����������������������������������������������������$�=�I�V�X�W�O�E�<�9�0�$���ƹƳƫƮƳ������������������������������������ �������#�(�����������������%�'�)�'���������ùìàÖÏÇ�z�w�t�zÇÓàìù��������ù�B�=�A�B�O�[�h�n�t�h�[�O�B�B�B�B�B�B�B�B���������\�@�M�Y�r�������ּ�������ʼ���	�������������������
�"�-�H�O�Q�M�;����������������������*�J�P�C�6�����߾�ܾʾľ����������ʾ׾��������������Ŷŵ��������������������������ƹܹ۹عӹܹ�������������ܹܹܹ�FcFVFAF=F1FFFF$F1F=FJFMFVF]FdFkFoFoFc���������� �����������������������������������������������������6�*�(�(�'�#�)�6�B�M�O�[�h�i�h�`�[�N�B�6�a�^�U�Q�U�U�_�a�n�v�zÀ�z�w�p�n�a�a�a�a���������������#�&�#�������ѿĿ����������������������Ŀѿ޿������5�����޿ٿݿ����5�Z�s���������s�N�5�����}�z�m�i�m�z�����������������������������������,�6�C�H�N�C�6�4�*�(�����ƳƦƟƧƳ�����������
�� �����������f�\�Q�L�O�\�c�h�uƁƃƍƑƚƜƚƎƁ�u�f�Ľ½ýĽǽͽнݽ��������ݽнĽĽĽ����
�	�
���#�#�)�-�#���������m�`�W�T�G�G�T�Y�b�m�y�}������������y�m���������������������	���&�'�"���	����������������������������	�
�������������������������
��#�0�4�6�0�'�#��
�������{�s�k�r�s�����������������������������<�:�5�<�H�U�W�V�U�H�<�<�<�<�<�<�<�<�<�<�U�b�g�n�~ŠŭŹ��������ŹŭŠŇ�{�n�b�U�6�0�6�6�B�M�O�V�T�O�B�@�6�6�6�6�6�6�6�6�s�g�g�]�Z�Z�Z�g�s�������������������s�s�����������������������������������������-�-�,�-�.�:�F�S�Y�Z�[�S�F�:�-�-�-�-�-�-�����w�m�n�������ɺ��
�	���غ�������ǈ�}�{�o�b�V�S�V�Y�b�o�{ǈǈǔǗǔǔǈǈ������ĿĻĿ���������������������������ĳĭġğģĵĿ�������������������Ŀĳ����z���������������������������������������ݽ������(�A�M�U�b�f�Z�M�4�(��0�$�������$�0�=�E�I�N�Q�P�I�A�=�0������������ �!�$�0�4�:�0�$�����������������������������������������������������������������ʾʾʾʾ������������������������������������������������׾վ־׾���� ������׾׾׾׾׾׾׾׾��۾־Ҿ������	�����
�������ù����ùϹ׹ܹ�ܹϹùùùùùùùùù����������������������������������������������������������Ŀѿݿ�����ݿʿĿ������������������������������������������D�D�D�D�D�D�EE7EPEVEbE\EZEPECE,EED�D�ùøìóù��������������������������ùù�<�:�/�*�*�/�<�H�U�a�_�U�H�A�<�<�<�<�<�<��ôàÓÎÐÇ�z�zÓìû���������������ż������ʼּ̼�������������ʼ����������ºɺֺ���ֺͺɺ�������������������������!�'�,�!�����������E�E�E�E�E�E�E�E�E�FFF$F1F1F,FFE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� S F b D O / a U 6 5 G 1 0 6 l V � N ` S U 2 C ( S B b ? � @ > F 0 ` [ J Q 6 D Q Y ) V . L ' q t ? P - - h $ 7 < P > 6 M J C 5 k < o K � M $ ; � j r X z j    �  �  �  �  �  y  �  �  �  -  �  �  0  �  �  �  �  �  P  �  �  �    �  
    �  )  �  �  �  2  �  �  �  w    �    �  �  @    �  �  E  n  P    E  �  �  �  y  �  �  Q  �  #  I    �  �  p  =  `  �  :  )  /  �  I     n  �  �  <t�<t�;o;�o�T������,1��t��\)��9X�u�u�P�`���t���C���C���9X�'��
�y�#�L�ͽ8Q�T���o��h�#�
��9X���ͽ��\)���@��ixռ��������\)�t��o���8Q�49X�D���\)�C��D���C��H�9�#�
�0 Ž�
=�H�9�ixսq���]/��j�}�q���aG��Y��]/�T���y�#�e`B�P�`����q���C���+����������
������1��񪽴9XB[B&�B��B�DBoWB�B�B�	BnB��B��B- LB*Z�B�eB�<BҗBd�B 2oB�B�=B ?%A�;B" B��B�B��B�B�8BTB�9B�B�7B�MA�E�B
�UBk�A�&�B.B)��B �4Ba�B�Br�B٦B�)B��B�rB
��B�VB�+B';BBK�B�hB
mKB ,�B&��B�`B��B	�qBq�B&��B�By�Bd�B%ފB��B#��B�B��B
�B	�6B-�FB!��B"l�B;�B8B�MB&��BBsB\�B@gB�lB��B˖BAbB�8Bg�B,�B*FB�]B�cB�~B>�B ?�B��B9eB 4�A�~�B��B�qB��B��B?iB�hB�XB��BIqB�B��A���B
�jB:�A�p�B{�B)ҽB ��BcjBD�B��B�=B�B��BM�B
��B��B�"B'@�B�BAYB��B
@�B ?�B&�8B��B��B	��B�B&@BD�B	�&BHKB%��BÅB#?�B@BB
��B
?�B. �B!AWB"=�B?|B �A�f@���A�j�A�;�Aŗ@��}@�GBB�A���A�6�A��Y@z�Au�A�^�B
d�B+�A��p@�zA�tVA�/@�A�l�A��AT�A��?.:C�� A�J A�EA�Q�A��3A�PrAw��A��'A�ڡA�.�B�TB�eA*��A��<Al*NA���A�}5A荚A��>A�i�A�y�A؊zA���A��@��@3�BHA�?�A� �A �A7�|B
cB	�A��AL]�A!g�AVO�AY�>�V�A��Ay��A�+�C�s�A�FDA�#tA̶RAV�@6�i@`��C���C�)�Á&@���A��iAЀBAŀ(@�ƣ@�PBB1A���A�j�A�V @y"�Au#JA��B	�/B$A��P@�D�A̅LAڀ�@��hA��A�rAS�A�]�?/�6C��A�u�A�~�A؇�AƃDA�� Ax�pA���A���A���B�JB5�A+�A��0AkhA�P�A���A��A�wA�<AA��A؝A���A�I�@|Kl@;�'B��A�`#A��\A �>A8�B
AB��A��AL�A!��AV!�AZ^�>~�A���Ay.�A��(C�XA�%DA�[wÃOA�@-i@g�wC��)C�"�                     �                  +               
         /   #      $                              %                  	                              	      E               5            	   
         	            T   
               	                           ?                  1   '   %                  -   +   #   !                           !   -                                                      +         !                                          #         '                                    '                  %      !                  #      !                              !   %                                                                                                                                 N���Nb��N�ޜN�[N�a�O$�wO�p�N��O#,CN�K�N���N�F�O�g�OV��O��BN���N�^ZN��O��N~��O��O���O���O>�\O0AxN�ʓOK;N	K�NzOA�"Nf*O��HOё�O�YN���O&�7Oj��OH��N�D�NNO��O�+�OT�ROe�N��[N6$O&�N��NB�;O	/NZ�O��Nc�N�]O��
N���O8+�O[�N��O�gN�r�N�c_N�hO���N#QCN6�OUT�N��NO�gjOo�N�7�Og�OCoN��N{�O�oN�/l  �  Q    �    <    �  �  �  �  r  �  �  a  d  e    �  �  �  ]    A  %  l  �  d  �  �  �  E  n  �    �    E  �    �  �  �  s  �  �  �    2  �  y    *  �  �  �  �  v  M  �    �  �  �  �  �    q  5  a  �  �    9  �  '  +<�o<e`B<e`B<o��o��o����t��#�
�49X�o�o���
��C��u�D���e`B�T���u�u�ě�����9X�����㼋C���t���t����㼛�㼴9X���
��1��/��1��1��j��j��j�ě����ͼ�/��/����h��h�\)�o���o��P��+�'<j�'#�
�P�`�'H�9�8Q�8Q�8Q�@��@��D���H�9�L�ͽT����+�ixսu�����o���㽝�-���������������������9<IU]ZUMIA<:99999999����������������������������������������������������������������������������������
��������GNO[_gntwztg[NJFGGGG��������������������,/<BHMOOH<<</.+,,,,,aanrvz{|{zna`XY\aaaa��������������������x��������������}spqx`aenz����������zve^`�����!+.)������������������������������������������������������������������������%$ ��������������������TV[amz��������zma[UT5BN[gji]NB5.--..,,.5�������
����������
!$$"
 �����xz���������zvrssxxxx����������������������	���������������������������������#/<HU_c`UTJ><8/-#@BFOY[^[PONB77@@@@@@-5B[gpt���yg[NB850+-��������������������PWar�������|qmaTPKKP���������������|u��������������������TY`mz��������zma]XST�����������������������������������������������������������@COVZ\_ht���thVODA@@it}��������������thi8<>Uan������znaTMH<8��������������������ot|���������tnjmoooost�����trnssssssssss������������������������������������������������������������������
���������EIPUXbfdbUIHEEEEEEEE������������������������������������������������������������egt�����������tidaae��������������������026<IUWZ\\[UUI<82000����
#+/(#
�������������������������S[git�������~tg[TPSSQUYanz|�zwnjaa_UNMQQ9<FIOU_YVUJIG<119999���)5BN[gt~���tgNB5)#S[hmkh`[YRSSSSSSSSSS"#04<HE<20,#""""""""r{���������������zr��������

���������  #&'#
���������	����������������������������JNX[it}������ztg[XNJ������������������������������������������������������!#+--2;?BC=:/#,/3<FHU]`YUH<71/.-,,àÕàäìù��������ùìàààààààà�ʼż����ʼּݼ���߼ּʼʼʼʼʼʼʼ�¦²³¿����¿¶²¦�����������������������������������������H�H�A�H�I�S�U�a�d�h�g�a�[�U�H�H�H�H�H�H�S�M�F�C�F�K�L�S�_�x�������������x�l�_�S���������'�4�M�f����������r�f�M�4�ƎƂƁ�~ƁƇƎƖƚƧƱƳưƩƧƚƎƎƎƎ�����������������
��#�/�1�2�/�,�#��
���"���"�(�/�;�H�R�T�T�T�J�H�;�/�"�"�"�"�A�A�;�A�N�Y�Z�g�s�}��v�s�g�Z�N�A�A�A�A�!���!�-�1�:�C�F�S�Y�S�S�F�:�-�!�!�!�!�����|�r�n�q�y�������Ŀؿ���ݿĿ������������������������������������������������������$�0�=�I�U�T�M�C�@�:�0�$���ƹƳƫƮƳ���������������������������������	�����#�%���������������������%�'�)�'���������ùìà×ÐÇ�z�v�x�}ÇÓàìù��������ù�B�=�A�B�O�[�h�n�t�h�[�O�B�B�B�B�B�B�B�B�g�Y�K�S�f�r��������Ѽܼ�ؼʼ��������g�"��	�������������	��"�/�2�;�B�@�;�/�"��������������*�C�H�C�6�*����������Ѿ׾̾ʾ��������˾׾�������� ��������Źż����������������������������Ź�ܹ۹عӹܹ�������������ܹܹܹ�FBF=F1F$FFFFF$F*F1F=FJFLFVF\FjFcFVFB���������� �����������������������������������������������������6�*�(�(�'�#�)�6�B�M�O�[�h�i�h�`�[�N�B�6�U�S�U�X�a�f�n�r�z�{�z�t�n�a�U�U�U�U�U�U���������������#�&�#�������ѿ����������������������Ŀѿݿ���ݿ��(���������5�N�^�g�s�{���{�s�Z�A�(�����}�z�m�i�m�z�����������������������������������,�6�C�H�N�C�6�4�*�(�����ƳƦƟƧƳ�����������
�� �����������f�\�Q�L�O�\�c�h�uƁƃƍƑƚƜƚƎƁ�u�f�Ľ½ýĽǽͽнݽ��������ݽнĽĽĽ����
�	�
���#�#�)�-�#���������m�`�W�T�G�G�T�Y�b�m�y�}������������y�m���������������������	���&�'�"���	����������������������������	�
������������������������
���#�0�.�#�"��
���������{�s�k�r�s�����������������������������<�:�5�<�H�U�W�V�U�H�<�<�<�<�<�<�<�<�<�<�x�s�{ńŔŠŭŹ����������ŹŭŤŠŔŇ�x�6�0�6�6�B�M�O�V�T�O�B�@�6�6�6�6�6�6�6�6���t�s�g�c�e�g�s�{���������������������������������������������������������������:�0�-�-�-�1�:�F�P�S�P�F�:�:�:�:�:�:�:�:�����������������ɺֺ����������ֺɺ�ǈǄ�{�o�c�c�o�{ǁǈǔǕǔǌǈǈǈǈǈǈ����������������������������������������ĳĮĢĠĦĸĿ����������	� ��������Ŀĳ����z�����������������������������������(������	���(�4�A�M�M�[�^�Z�M�A�4�(�0�$�������$�0�=�E�I�N�Q�P�I�A�=�0��	���������
�����$�'�$�!��������������������������������������������������������������ʾʾʾʾ������������������������������������������������׾վ־׾���� ������׾׾׾׾׾׾׾׾��۾־Ҿ������	�����
�������ù����ùϹ׹ܹ�ܹϹùùùùùùùùù����������������������������������������������������������Ŀѿݿ�����ݿʿĿ������������������������������������������D�D�D�D�D�D�D�EE7ECEPETETEJE9E'EEED�ùøìóù��������������������������ùù�/�,�.�/�<�H�U�[�]�U�H�<�/�/�/�/�/�/�/�/ùøìêàÕÓÓØàìù������������úù�������ʼּ̼�������������ʼ����������ºɺֺ���ֺͺɺ�������������������������!�&�+�!������������������E�E�E�E�E�E�E�E�E�FFF$F1F1F,FFE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� S + b D Q / ; 8 6 7 G 1 + 9 f V X N \ S K 3 K   M B T ? � @ @ F . T [ J Q 6 D Q Y ) V - L ' _ t = P + & ]   5 < B > < M J C 5 k < o K � P $ 9 S j r I z j    �  s  �  �  �  y  C    a  �  �  �  (  �  �  �  �  �  B  �  �  #  �  �  �    l  )  �  �  |  2  �  �  �  w    �    �  �  @    M  �  E  {  P  _  E  h    ~  �  S  �  �  �  �  I    �  �  p  =  `  �  :  �  /  �  �     n  �  �    =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  �  �  }  r  f  b  _  Q     �  p  I  "   �   �   �  C  C  D  E  J  O  J  =  0  !      �  �  �  �  }  _  ?              �  �  �  �  z  T  .    �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  y  q  i  b  Z  S  K  5     �   �   �            �  �  �  �  a  �  �  t  &  �  �  K    �  g  <  5  *        �  �  �  �  �  �  �  �  j  <    �  �  �  
�  .  �  �  ;  n  �  �    �  �  [  �  t  
�  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  K  "  �  �  w  8  �  �  �  �  �  �  �  �  z  J    �  �  N  
  �  z    �  �  d  r  |  �  �  �  �  �  �  u  a  F  '    �  �  ]    �  4  �  �  �  �  �  �  {  i  V  B  .      �  �  �  �  �  �    r  h  ^  T  K  B  7  ,  !      �  �  �  �  �  �  �  �  t  =  g  {  �  �  �  �  �  |  c  <  	  �  �  H  �  {    b   �  z  �  �  �  �  �  �  �  �  }  j  Q  5    �  �  �  d  
   �  C  Y  a  Z  J  5    �  �  �  v  >  �  �  p    �    k   �  d  _  Y  S  L  >  0  !    �  �  �  �  �  �  e  A     �   �  J  O  U  [  a  c  Z  Q  H  ?  4  (         �   �   �   �   x    �  �  �  �  �  �  �  �  �  �  �  z  d  J  "  �  v  W  9  �  �  �  �  �  p  H    �  �  �  �    �  �  o    �  	    �  �  �  �  �  �  �  �  �  �  �  �  u  \  C  *  �  �  N    m  �  �  �  �  �  �  �  Q     �  e  �  �  �  e    �  g  o  �  �  �    $  9  J  X  \  W  H  ,    �  �  m  5  �    \  �  �           �  �  �  �  �  �  ]    �  A  �  �  �  '  Y  �  �  �    )  8  @  >  0    �  �  �  X    �  =  �  �  �  �         �  �  �  �  �  _  ?    �  �  �  b  %  �  �  l  `  Q  @  /  %  .  "    �  �  �  �  p  ;    �  �  Z    �  �  �  �  k  I  $  �  �  �  n  :    �  z  )  �  u  �    d  `  ]  Z  V  S  Q  N  K  I  F  C  ?  <  8  1  )  !      �  �  r  c  b  c  e  T  :     �  �  �  �  [  /     �   �   t  �  �  �  �  �  �  �  �  c  C     �  �  �  r  6  �  m    �  G  s  �  �  �  �  s  ^  I  1    �  �  �  �  �  w  (  �  z  E  7  +      	  �  �  �  �  �  �  �  �  z  Z  9       �  b  m  h  Z  @      �  �  �  �  �  �  V  �  f  �  �  x  (  R  k    �  �  �  {  w  �    q  V  ,  �  �  �  R  �  Q            �  �  �  �  �  �  �  �  �  �  �  �  �      6  �  �  �  �  �  �  �  �  l  R  7      �  �  �  ~  W  .        �  �  �  �  �  �  �  �  �  k  H  "  �  �  �  `    �  E  A  9  *    	  �  �  �  �  �  �  �  |  k  Y  H  3       �  �  �  �  u  b  P  3      �  �  �  �  x  L    �  �  T    �  �  �  G  	            T  p  �  &  �  �  �  �  e  �  �  �  �  �  �  }  u  m  c  Y  O  @  .    	   �   �   �   �  �  �  �  �  �  z  l  b  U  ?  &    �  �  �  ~  K    �  �  �  �  �  �    ^  <    �  �  �  �  �  |  c  K  2    �  �  L  m  r  n  h  _  S  E  0    �  �  �  ]  %  �  �  �  f  L  �  �  �  �  �  �  �  �  �  }  u  m  d  [  Q  G  =  4  +  "  �  �  �  �  �  �  {  u  n  g  _  X  P  H  ?  7  +        �  �  �  �  �  �  �  �  �  �  �  �  k  S  <  "    �  �  �        �  �  �  �  �  �  �  �  �  �  �  �  w  g  X  I  :  �  �  �      $  -  1  -  !  �  �  w  8  �  �  c    �  p  �  �  �  �  �  u  [  A  %    �  �  �  �  �  �  �  x  1   �  r  u  w  v  j  _  N  :  &    �  �  �  �  �  }  `  A  !     �  �  �  �  �  �          �  �  �  �  m  �  ;  Q  B  �      �    '      �  �  �  l  4  �  �  �  E    �  �  d  �  �  �  �  �  �  �  �  �  �  �  �  �  l  O  2    �  �  �  �  �  �  �  �  �  s  c  N  6    �  �  �  �  q  G    �  �  �  �  �  �  �  v  _  C  "  �  �  �  P    �  \  �  �     �  �  -  _    �  �  s  \  ?    �  �  K  �  �    �  %  X  �  v  d  O  6    �  �  �  �  R    �  �  V    �  ]  �  P   �  �  �         2  H  M  =    �  �  k  �  q  �  X  �  p  W  �  �  �  �  r  b  P  ;  "    �  �  �  x  U  3    �  �  r       �  �  �  �  �  �  �  �  �  �  w  r  n  [  C  '    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  g  Z  M  A  5  )      �  �  �  �  �  u  a  G  0  !  /  \  p  c  O  :  $    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  t  m  f  _  X  P  I  B  ;  4  ,  %              �    �  �  �  �  �  �  �  y  U  *  �  �  �  <  �  |    �  �  q  k  f  ]  N  @  /    	  �  �  �  �  �  c  C  #  �  �  t  y  �  '  5  '    �  �  8  �  s  �  
�  	�  	  �  �  ;  �  �  a  V  L  A  6  (    
  �  �  �  �  �  �  i  N  1    �  �  �  �  �  �  �  �  �  �  p  [  <    �  �  o  ,  �  �    �  Z  �  i  O  a  �  �  p  K  *    �  �  �  �  Y  (  �    �    j  R  :       �  �  �  �  �  e  A    �  �  �  D     �  9  7  4  2  /  -  *  ,  0  5  9  =  B  ?  .      �  �  �  �  �  �  �  �  �  �  u  c  R  >  )      �       2  m  �  '    �  �  �  S    �  x  H  E  7      �  .  �  /  �  �  +        �  �  �  �  �  �  �  i  K  *  �  �  �  �  �  }