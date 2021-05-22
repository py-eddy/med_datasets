CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���
=p�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       MȻ�   max       P���       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�j       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��\(��   max       @F�33333     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��   max       @v�Q��     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @P�           �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�J            5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��S�   max       <�t�       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��u   max       B5       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4��       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��Y   max       C��Y       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�   max       C��p       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          A       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MȻ�   max       Py�       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?�|����?       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <�j       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @F�33333     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��   max       @v�Q��     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P�           �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�k�           V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?}��,<�   max       ?Є���?     P  X�                     A                                    -                           	   	                        $                           $            '   '                        !      ,            O�S�NK��O�O-� O�&{NEA.Pf@�O�;�N�GPo'DO6�O���O��N��N�
�Np=�O���NL-P���N���O�7dOubLNɞ OY�fObO�Ot�N�N�B�N�'{N_��Oa�NmKOk�}N��EOvUzO���P�Oˆ�O�6�N��8NMO ��MȻ�N�6/N��P �N���Ov�GO��O��.PP$O�u�O��GOC�N[Y9NbǳOaN��2Ows�N��O��Na�N���O7 �M��<�j<�j<��
�D���ě��o�o�t��#�
�49X�T���u��o��C���C���C���t���t���t����㼣�
��j��j��j�ě��ě�����������/��/��/��/��`B��h��h�������o�o��P��w��w�#�
�#�
�'0 Ž0 Ž8Q�8Q�H�9�L�ͽL�ͽP�`�aG��ixսq���u�}󶽋C���C���hs��hs�������������������������������(.0)%!	
gnz����������zy{zrgg&/9HU_glliaUH</#������������������������/0!0<bn~nGH/'
������������������|{�)-56852)bm{��������������mcb)67>C@96.)'�����������������������������������45BNR[[][NIB=5444444ggt���������}wtolggg��������������������)046BOX_h|}zth[OB.&)NO[\ghkh[XQONNNNNNNN�
#b������{U<����������������������),7B[hz�����tgD5/)*)��������������������������

���������N[ahnz������~znaZURN������������~|{{|��������������������KN[ghmkgg\[NNCCFKKKK�����������������������������������������������������������)57851)	������������������*/<HSUafnryyvnaUH:.*����������������������������������������T]amz~���~zwxqaTQORTF[t���������tfnhcUGF���&67@=90#������mz�������������znihm��������������������mt����tplmmmmmmmmmm����������������������������������������������������������������������������������������� #�������������������������jnv{���������{iedgij~�����������������~������������5BNZZML5)�����al�������znaULF=@HUaNgt�����������tg[SNN������������������|���������������������!#$)+./<?<8992/$#"!!FHUadgjknqna\UNF?<?F��������������������v}��������������~wtv���������������������)4:>?7)��������������������������7<>DIKMMMMJI<9217777`grt����������trgc]`

�g�S�[�_�g�t¦²¿����½³¦�������	���� �������������ɺź��ɺֺ�������!�$�!��	������������������������������
�����������ҾZ�M�A�<�4�-�.�4�A�M�Z�f�y�x�t�k�k�m�h�Z�#�!�����#�/�/�:�2�/�#�#�#�#�#�#�#�#�����i�a�m²�����������
�$�8�#����)�(�(�*�%�)�;�B�L�[�h�n�x�t�h�[�O�B�6�)�=�4�0�-�+�0�<�=�I�P�V�[�V�U�I�@�=�=�=�=Ƴƅ�h�H�>�?�N�T�h�uƎƧ�������� ����Ƴ��x�s�Z�T�N�L�M�Z�f�s����������������ìàÓÎÆ�~�{ÂÆÓàæìÿ��������ùì�U�a�n�q�s�r�n�e�a�U�H�G�=�<�6�:�<�H�T�U�����������������������������������������Ŀ������Ŀοѿݿ�����������ݿѿſĿľ������	���	������������𿒿y�`�O�I�L�T�`�f�y���������������������Y�W�Q�Y�e�f�r�s�u�r�e�\�Y�Y�Y�Y�Y�Y�Y�Y�����������z�H�5�)�A�T�����������������俟�������������Ŀǿ˿ĿĿ����������������վʾ������׾����	��!�"�!���	�����;�/�*�'�$�/�0�;�H�T�a�e�g�d�\�T�L�H�=�;ŭţŠśşŠŭųŹ����������������Źŭŭ�������������������������	����	����𾘾��������ʾѾ޾������׾ʾ��������M�E�A�9�9�A�M�Z�f�s���������������f�M����������)�)�6�7�6�6�)����������������*�1�0�-�*����������������������$�'�+�)�$�����5�5�2�5�A�E�N�Y�Y�N�J�A�5�5�5�5�5�5�5�5�g�`�\�Z�]�g�z�����������������������s�g��Ʒ��������������������������������������������������	��"�(�-�,��������
�����
����*�4�4�*�&������Y�Q�@�'����!�/�4�=�@�M�Y�f�j�q�u�k�Y�L�5�-�$�$�(�5�A�N�Z�]�g�s���������s�Z�L���z�r�w��������������1�9�"����������� ����лǻûлܻ����!�0�7�6�@�'�� ��ĻĽ��������#�5�<�G�<�0�$��	���������U�I�L�U�\�b�n�{�{�{�r�n�e�b�U�U�U�U�U�UƚƕƚƞƧƳƹƶƳƧƚƚƚƚƚƚƚƚƚƚ�Y�T�T�S�V�Y�e�r�~�������������~�r�e�Y�Y���������������������������������������������������Ŀѿݿ޿����������ݿѿĿ��������y�m�d�`�\�`�m�y��������������������������u��������ּ��!�*�1�.�!��ּʼ����������(�(�5�:�A�H�A�>�5�(�����������~���������û������ܻлû�����ŭŧŠŔŇ�~ŇŔŠŭŹ������������ŽŹŭ�f�Y�L�=�;�@�M�Y�f�����ʼּϼ��������r�fÇ�n�]�P�O�F�J�U�aÓù������������ùàÇ�������Ŀѿݿ�ڿۿ���޿ѿĿ��������������������������������"�%��������¬¦¢¤¢¦²³¿������������������¿¬�x�n�l�h�l�x�������������x�x�x�x�x�x�x�xE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFF$F1F=FEFJFRFNFJFEF=F1F$FFE���������������������������������/�#����	��
�#�/�<�U�a�b�c�a�U�H�<�/�Ľ����Ľнݽ�ݽܽнĽĽĽĽĽĽĽĽĽĽY�C�:�5�8�N�l�����������������������l�Y���������
����������������������������ݽ׽ݽ������������������ĳĪĦĦĤĦħĭĳĿ����������������ĿĳE*EE'E*E7ECEFECE<E7E*E*E*E*E*E*E*E*E*E* J 1 c > E ( X + , L M V > f Q 9 0 [ N K G ? 9 S + O - + F Q = h ? J - o � F K ^ A , J A V e U S @ J ? b $ + E � ^ < : N 4 H g 4 ]    =  f  �  �  )  Y    -  �  7  �  /  k  �  �  s  �  D  �  �  7    �  �  �  *      �  x  �  �  �  �  �  f  �  �  \  �  7  b  
    	  �  �  	  ]  l  �  �  L  �  e  �  �  �  �  5  @  1  �  �  ��o<�t�$�  �e`B�C��D����t��\)��t��,1��`B�0 Ž�P��9X��9X���
�#�
���ͽ}�ě��49X�#�
��h���#�
�#�
���\)�\)��h��P�+�'C��]/�<j��7L�e`B�e`B��P�#�
�T���'@��<j���-�aG��aG��Y���1��-��������7L�y�#�}󶽓t�����\��hs��S����P���-��vɽ�9XB��B�YB��B��B�fB��Bf�BbxB�pB-B/�Bd�B!B_B��B
	�B�B�~B>�B&��B�B��B�ZB͏B�B� B"2�B��B^"BWB�GBB�+BϓBҺB ��A��uB[B$CeB �BIZB	�)B!oZB5B*hB+�B-/�B�jB(��B)LB9�BO�B�'B

�BJ�B!�B5<B�uB"��B
��B��BX�B1�B&��B
BOB�B�B�zB��B��B�B%�B=�B�B��B @2B�0B�>B ޟB�nB
ZkB;@B�[B@�B&��B/�B	)�B�%B�#B�BA1B"h�B��B��B=B[�B��B��B�^B�oB ��A���BĈB$@�B ��B@oB	�JB!@B4��B)�/B+-�B,�	B8�B)=jBE^B��B��B@�B
?=B��B �B;+B��B"�=B
�DB@8B?�B,�B&K�B	��B�}A��A�53@E:�A�A<]A���A�gA��B
�+Bw�AD�A�g�A��\Ar�bA|V3AX׹Ao+�?��YA�
1Av.�AWQ�A���A�?A�'�AO�{A@�MA�CEA��B	rA���A��cB�HA[A�d�@�D\A���A���@���A�bAB��?�%gAM�A|�Am̓A �A�$@���A�x@���A�tPAx�A���A���@���C�MiC��Y@Z�fA�\�A(~�A�
B��A0[*A┇C��>A�qA�}�@C��A�z�A<��A��uA���A��B;>B>�AECA�_AŅAq=�A}AX�1Ap��?�A�wAwh�AY"tA��=A��XA�zuAO��A?H$Aէ�A�j�B	FA�
A�^�B�0AZ�EA�m@٫~A��HA��@�c�A�|�A�`�B�B?� �AM=A|��Am�HA �>A��e@� {A��J@��A�z	Ay�A��"A�� @��#C�9�C��p@\dcAÅ\A'
A�+BLA2�gA�>fC��                     A      	                               -                           
   	                        %                           %         	   (   '                        "      ,                                 E         9                     !      =      )                                                /   '   %                     5            '   -                               #                                          9                     !      7                                                            #                     1            #   -                              #            O?�NK��N��O-� N���NEA.O��O�2N�GPo'DO6�OX{jO�N��N�
�Np=�O���NL-Py�N���Oo~OubLNɞ OY�fObO�OFb�N���N��[N�2N_��Oa�NmKONK9|Oj/iOlchO�o�O �jO�VN��8NMO ��MȻ�N�6/N��P0�N���N��O��O��PP$O{�@O,�OC�N[Y9NbǳOaN��2N��BN��OŒ+Na�N���O7 �M��  �    �  �  l    y  "  f  V  �  �  �    �  �  W  �  V  �  �  �  C  8    t  �  �  "  X  �  �  �  m  �  �  d  �  �  h  <  /  l  �  +  �  �  8  ^    n  g  �  L  X  �  2  f  �  h  �  e  �  R  }<�t�<�j<�C��D�����
�o�C�����#�
�49X�T����t���C���C���C���C���t���t����
�����h��j��j��j�ě�������/��/��`B��/��/��/�o�����+�''+�o��P��w��w�#�
�#�
�0 Ž0 Ž@��8Q�L�ͽH�9�P�`�Y��P�`�aG��ixսq���u���-��C������hs��hs��������������������������������$)+.*)"
	gnz����������zy{zrgg>HU_aca\UHF=>>>>>>>>�����������������������
#/HH<8*#������������������������)-56852)bm{��������������mcb)67>C@96.)'�����������������������������������45BNR[[][NIB=5444444ggt���������}wtolggg��������������������)046BOX_h|}zth[OB.&)NO[\ghkh[XQONNNNNNNN	#?Un������{U<
�	��������������������?BNS\gtz����tg]NG;;?��������������������������

���������N[ahnz������~znaZURN������������~|{{|��������������������NN[ggljgf[NEDHNNNNNN�����������������������������������������������������������)57851)	������������������=HUZainprnnaUQHC====����������������������������������������T^amz��zuurnaTRQPQTKP[hty�����tmh[VPLK���� 
#)&#
����mz�������������zojim��������������������mt����tplmmmmmmmmmm��������������������������������������������������������������������������������������� "���������������������������mnq{���������{wnkikm~�����������������~����	��������5BNZZML5)�����BHUakn�����znaULGD?BXgt�����������tg][VX������������������|���������������������!#$)+./<?<8992/$#"!!FHUadgjknqna\UNF?<?F��������������������~�������������~~~~~~���������������������)256996)��������������������������7<>DIKMMMMJI<9217777`grt����������trgc]`

¦�g�g�k�t¦²¼»¸²¨¦�������	���� ������������ɺȺĺɺҺֺ�����������ֺɺɺɺ�������������������������
�����������ҾA�5�5�A�H�M�Q�Z�a�[�Z�M�A�A�A�A�A�A�A�A�#�!�����#�/�/�:�2�/�#�#�#�#�#�#�#�#¿²ª¨©­������������������������¿�6�4�1�.�2�6�B�O�Q�[�c�h�j�h�e�[�O�B�6�6�=�4�0�-�+�0�<�=�I�P�V�[�V�U�I�@�=�=�=�=Ƴƅ�h�H�>�?�N�T�h�uƎƧ�������� ����Ƴ��x�s�Z�T�N�L�M�Z�f�s����������������ìàÕÓÐÁ�ÆÇÓàáìùü������ùì�H�G�>�<�7�;�<�H�U�a�n�p�s�q�n�m�d�a�U�H�����������������������������������������Ŀ������Ŀοѿݿ�����������ݿѿſĿľ������	���	������������𿒿y�`�O�I�L�T�`�f�y���������������������Y�W�Q�Y�e�f�r�s�u�r�e�\�Y�Y�Y�Y�Y�Y�Y�Y�����|�g�S�C�=�Z���������������������������������������Ŀǿ˿ĿĿ����������������ʾǾƾʾ׾�����	�������	��׾��;�/�*�'�$�/�0�;�H�T�a�e�g�d�\�T�L�H�=�;ŭţŠśşŠŭųŹ����������������Źŭŭ�������������������������	����	����𾘾��������ʾѾ޾������׾ʾ��������W�M�H�C�;�>�A�M�Z�f�s�z���������s�f�W���������'�)�6�5�)�����������������*�0�.�+�*��������������������$�&�*�'�$�������5�5�2�5�A�E�N�Y�Y�N�J�A�5�5�5�5�5�5�5�5�g�`�\�Z�]�g�z�����������������������s�g��Ʒ����������������������������������������������	��"�"�&�&�"���	������������*�1�/�*�$����������Y�S�@�'����"�0�4�?�@�M�Y�f�j�q�t�j�Y�R�5�0�'�(�5�A�N�V�g�s�������������s�g�R�������~�|�������������������������������������ܻܻ��������&�'�����ĽĿľ����������#�1�;�0�#�����������U�I�L�U�\�b�n�{�{�{�r�n�e�b�U�U�U�U�U�UƚƕƚƞƧƳƹƶƳƧƚƚƚƚƚƚƚƚƚƚ�Y�T�T�S�V�Y�e�r�~�������������~�r�e�Y�Y���������������������������������������������������Ŀѿݿ޿����������ݿѿĿ��������y�m�d�`�\�`�m�y���������������������������������ּ��!�'�/�.�!����ּʼ����������(�(�5�:�A�H�A�>�5�(�������������������»ûлܻ��ܻ׻лû���ŭŧŠŔŇ�~ŇŔŠŭŹ������������ŽŹŭ�f�Y�P�B�?�@�H�Y�r��������Ǽ�������r�fÇ�n�]�P�O�F�J�U�aÓù������������ùàÇ���������������Ŀѿؿڿ���ݿѿĿ�����������������������������!�������¬¦¢¤¢¦²³¿������������������¿¬�x�n�l�h�l�x�������������x�x�x�x�x�x�x�xE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFF$F1F=FEFJFRFNFJFEF=F1F$FFE���������������������������������<�1�/�#�"�#�#�/�<�H�Q�U�H�C�<�<�<�<�<�<�Ľ����Ľнݽ�ݽܽнĽĽĽĽĽĽĽĽĽĽ`�F�:�>�S�l�y���������������������y�l�`���������
����������������������������ݽ׽ݽ������������������ĳĪĦĦĤĦħĭĳĿ����������������ĿĳE*EE'E*E7ECEFECE<E7E*E*E*E*E*E*E*E*E*E* @ 1 , >   ( : ' , L M W @ f Q 9 0 [ K K S ? 9 S + F . - < Q = h 6 S . r ] 1 C ^ A , J A V m U 1 @ @ ? S   + E � ^ < % N 3 H g 4 ]    �  f  �  �  �  Y  i  ?  �  7  �  �  S  �  �  s  �  D  7  �       �  �  �  �  �  �  �  x  �  �  7  {  �  3  m  e     �  7  b  
    	  	  �    ]  �  �  7  �  �  e  �  �  �  �  5  �  1  �  �    >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  q  �  �  �  �  �  �  u  Z  9    �  �  y  H    �  (  \            
       �  �  �  �  �  �  �  �  �    j  T  ?  3  �  �  �  �  �  �  �  �  n  O  -    �  �  ]  �  x  �  i  �  �  �  �  �  }  �  �  �  �  �  }  v  b  I    �  �  t  C  �  �  �  �  �  �    ?  U  i  e  L  ,  �  �  p    �  �       �  �  �  �  �  �  �  �  �  x  f  T  B  0    �  �  �  �  �    6  h  �  �    n  x  q  Q  ,  �  x  �  2  w  n  �  &  �  �  �  �        !      �  �  �  h    �  "  �  I    f  f  e  \  L  ;  %    �  �  �  �  �  j  I    �  R   �   �  V  F  B  :  .      �  �  �  h  <  
  �  �  t  '  �  �  &  �  �  �  �  �  �  �  ~  k  V  8    �  �  �  �  s  I  G  �  �  �  �  �  �  �  �  _  6  
  �  �  u  '  �  m  "  W  c   W  �  �  �  �  {  _  =    �  �  ~  I    �  �  �  �  Q  �  $      �  �  �  �  �  �  �  �  �  x  `  D  (    �  �  �  �  �  �  �  �  }  o  a  S  C  1       �  �  �  �  �  g  9    �  �  �  �  �  �  �  �  �  �  z  s  l  ]  >     �   �   �   �  W  I  =  7  7  4  $    �  �  �  �  n  6    �  �  �  F   �  �  �  �  �  �  �  �  �  �  �  z  b  I  0    �  �  �  �  �  K  T  <  4    �  �  �  g  7    �  �  K  �  �  1  �  g   �  �  �  �    v  l  b  X  O  F  =  4  )        �  �  b  )  �  �  �  �  �  �  �  �  �  �  �  �  r  F  1    �  �  7  �  �  �  �  �  �  �  �  �    n  W  <    �  �  n    �  \   �  C  :  1  (        �  �  �  �  �  �  �  �  �  �  �  �  �  8  ,         �  �  �  �  �  �  �  �  �  o  Z  E  3  !             �  �  �  �  �  �  �  �  �  i  R  9    �  �  �  L  `  o  r  n  e  W  C  )    �  �  �  �  d  :    �  �  �  �  �  �  �  �  �  �  �    �  �  r  a  M  3  	  �  �  W    �  �  �  �  �  �  �  �  �  ~  i  S  <  '      �  �    "      !        �  �  �  �  �  �  �  �  j  ?    �  �  y  X  S  O  J  E  @  ;  7  2  -  #       �   �   �   �   �   �   �  �  �  �  �  |  c  H  *    �  �  �  �  �  �  �  �  �  m  S  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  j  W  C  �  �  �  �  �  �  �  �  �  �  �  s  `  J  (    �  �       \  `  e  j  l  f  `  Z  Q  E  8  +    �  �  �  I   �   �   I  �  �  �  �  �  }  R    �  �  n  2  �  �  �  (  �  j    �  }  �  �  �  }  u  i  W  @  #    �  �  a    �  �  0  �  -  �  �      $  H  b  Z  J  *    �  �  y  7  �  l  �  �  (  �  �  �  �  }    �  �  �  �  x  e  W  C  '  �  �  ?  �  S  �  �  �  �  �  �  w  R  (  �  �  �  b  /  �  �  �  b  �  �  h  _  V  L  B  3  $      �  �  �  �  s  U  6  3  6  9  <  <  5  /  (  !                 �  �  �  �  �  �  �  �  /      �  �  �  �  �  �  �  x  `  K  @  2  !    �  �  �  l  f  a  \  W  R  M  H  C  >  7  /  '           �   �   �  �  �  �  �  �  �  �  �  �  n  [  I  4     	  �  �  �  �  y  +  &  !            	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  M  ,    �  �  }  I    �  �  G  �  �  2  �  �  �  �  �  s  a  K  4       �  �  �  �  e  <    �  +  �  )  +  ,  .  1  4  7  7  5  )    �  �  �  �  k  B     �   �  ^  ]  [  P  @  /      �  �  �  �  [  3    �  �  �  l  0    �        �  �  �  �  x  .  �  {  >  �  �  8  �    �  n  ^  O  ;  %    �  �  �  �  �  W  *  �  �  �  V    �  M  S  d  R  F  F  F  <  /  +  4  2    �  �  �  U    �  �  n  �  �  �  �  �  �  �  �  n  H    �  �  �  @  �  �    �  �  L  4    �  �  �  �  L  '  2  A  9  '  	  �  �  t  ;  �  �  X  O  G  >  3  (        �  �  �  �  �  �  �  �    9  \  �  �  �  �  �  w  X  :    �  �  �  �  [  5     �   �   �   �  2  '  !        �  �  �  �  �  w  Q  %  �  �  z  =  �  �  f  ]  T  K  B  9  0  '          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  D  �  b  �    d  h  `  X  P  H  A  9  ?  L  Y  f  t  �  �  �  �  �  �  �  �  Y  �  �  �  �  {  e  J  (  �  �  �  J    �  H  �  U  �    e  =    �  �  �  u  n  v    �  �  �  �  Y  �  q  �  	�  
  �  �  �  �  �  �  �  l  X  C  .      �  �  �  �  o  Q  4  R  A  /      �  �  �  �  {  Y  4    �  �  �  ]  "  �  �  }  q  e  Z  N  C  8  2  .  +  )  (  '  !      �  �  �  �