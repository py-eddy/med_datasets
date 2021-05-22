CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�^5?|�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P��n      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���P   max       =��m      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���R   max       @F���Q�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(�    max       @v�          	�  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P`           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�          �  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ix�   max       >�C�      �  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��u   max       B3 �      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�l�   max       B2ȝ      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�\�   max       C��      �  7�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�S   max       C���      �  8�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          1      �  :�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  ;�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       Pu;�      �  <�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*0U2b   max       ?��D��*      �  =�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��t�   max       >#�
      �  >�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���R   max       @F���Q�     	�  ?�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�H    max       @v�          	�  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @P`           |  R�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           �  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�C,�zxm   max       ?�ѷX�       T�                     	            G               j   %                           ]   ,                  $   J   	   I   �                        $   1      2      .          �         	   x   	      
   'N��~N2G�NL<?N(�OYɞN��N��O{6FN0'�N�h5P��O �N (N�N;<2P��nOs�6N�|�O���OP�'N�#N�&O]\FNVABO?4Py�KP�NTOO}�O�O�N��;O^VbP �_PNq�OQ�O��Pk(O�N�O�%N�2�O���N� �O~�O��jP��OUߔN7��O��6N�CP��N�:rO�"�O癕N�dO"YN�@P$=�N�y�N�w}N'|�O,�Z���P��`B��1�u�T���t��ě��o��o;o;�o;�`B<#�
<e`B<e`B<u<�o<�C�<��
<�j<ě�<���<�`B<�h<�<�<�<�=o=+=+=+=+=C�=C�=t�=t�=t�=t�=�P=#�
='�='�=,1=,1=49X=8Q�=D��=T��=q��=y�#=}�=�7L=�C�=�C�=�C�=��P=���=�9X=��m=��m "#/3<AE@</#      ��������������������w|����������wwwwwwww����������������������	5BNYRNKA5)�MKNO[\g[UOMMMMMMMMMMroqt���������trrrrrr������������������������������������������������������������������#<;=<:0#
���bbdhhru���������xuhbABCO[d^[OBAAAAAAAAAA:2/1<HNU]abaURH<::::WT[_hnonh[WWWWWWWWWW���������������������������������c`afht�������~thcccc
#0<IU\[UI30#
��������������������������



������������� ����������TPRV[[ht��������thbT��������������������������
���������5N[dmqmbNB5�������#/1/(�������zwwvz}�������zzzzzzz�����������������#)1N[gtxxxng[NB;5,��������������������LIHLO[ht�����xth[SOL�������������������������)BJKH>6)�����#'.340+%#
�����������������������)N[orn[VI5)������&788BEB5)���IEELN[gtv{zvtkgb[PNI96:<>HU[`^UQH<999999��������������������$")*5@BNTVNB5)$$$$$$�
#/<N^fea^UH<,
����5BNZZNLB)��\gu��������������tg\��������������������	)-))
B[_hshd[D6)

!#&&''#



	
)5BNUWUPD:5)
_UU]ajmz�����zzmga__&)6BO[bc`[OB6)��������
 !
����
##+/-+#
��������
��������*+6:>61*)nvz���������������zn��
 "
	�������	
##)&#!
�������������������������

���������D�EEEEE EEEED�D�D�D�D�D�D�D�D�D������ڼּʼļ������ʼּ������㻷�ûͻлܻ�ܻлû��������������������������������������������������������������T�`�i�m�u�v�p�n�m�`�Y�T�G�@�@�C�F�G�N�T�y�������������{�y�x�y�y�y�y�y�y�y�y�y�y��������������������������������������������'�*�:�@�J�Q�L�I�@�3�'���������²½¿������¿²¦¦©²²²²²²²²E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E;���(�4�?�F�F�A�(�����ٽǽ����̽��ʾ׾��������������޾׾Ҿʾɾƾʿݿ������ݿܿڿܿݿݿݿݿݿݿݿݿݿ������������������������������������޻-�:�D�F�P�F�:�-�"�%�-�-�-�-�-�-�-�-�-�-Óàù������øùìÓ�a�H�:�4�3�>�U�nÁÓ�(�4�A�M�Z�b�e�m�s�x�r�f�Z�M�A�0�!���(�@�L�Y�e�i�q�o�i�e�Y�V�L�A�@�=�?�@�@�@�@���������������������l�\�M�I�C�M�Y�f����Ľнݽ��ݽнĽ����������{�����������S�_�j�l�x�~�z�x�u�l�i�b�_�S�P�K�O�N�S�S�����������ûϻƻû������������������������������ʼʼԼмȼ�����������������������������������������������������������������������������������������������#�I�X�n�x�y�v�n�b�U�0������������ؿ�����*�5�5�7�����ѿĿ������¿ѿݿ�F$F1F=FJFPFJF=F7F1F$FFFF!F$F$F$F$F$F$������������ �%�������������żŻſ�ƿ�����8�:�0�����ݿٿֿֿؿݿ����6�A�B�H�B�A�?�6�)�$�$�)�,�5�6�6�6�6�6�6�����������������������p�f�d�b�j�s�y��"�.�5�9�3�6�3�"��׾ʾ�������������	�"��"�/�;�M�S�P�@�"��	����������������������!�.�2�:�K�:�.�!�����������ĚĦĳĿ��������ĿĳĦč�t�h�[�c�kāčĚ�)�O�Y�Z�^�l�k�[�B�6�,���������������)¿����������²¦�t�q�v¦²¿�B�N�[�g�k�h�e�\�[�N�B�5�1�)�'�)�.�5�8�B����"�)�4�4�)������� �������5�A�N�Z�g�r�}���s�Z�N�5�(������(�5��(�2�4�8�4�3�*�(�������������"�'�/�2�0�+�"��	���������������������������������������������������������������� ��� �#�"���������������������àìù��������üìÞÓÇ�}�y�yÃÇÓÚà��'�*�3�4�=�3�'�'������������~�����������κ��������������r�b�]�b�k�~�m�m�o�m�`�]�T�G�;�.�"�.�9�;�G�T�`�j�m�mƳ��������������������ƧƎ�u�h�[�W�sƎƳ�{ŇŔŠťŭůŲŭŤŠŜŔŇŇ�|�{�z�{�{�����!�4�=�I�N�K�B�4�'�����������DoD{D�D�D�D�D�D�D�D�D�D�D�D�D�DsDhDhDnDoǔǡǭǲǴǭǭǡǔǈ�{�q�z�{ǈǐǔǔǔǔ�y�|���������������������}�y�v�t�o�s�u�y�l�y�{�y�q�m�l�e�`�S�H�G�F�G�N�S�`�k�l�l���ûŻлܻ�����ܻл������������������n�t�{�{��{�n�b�U�L�U�U�b�i�n�n�n�n�n�n�6�C�O�R�X�O�O�C�6�,�*�#�*�+�6�6�6�6�6�6E7ECEPEZE\EbE\EPECE?E7E3E7E7E7E7E7E7E7E7E�E�E�E�E�E�E�E�E�E~EuEqEjEiEcEiEuE�E�E� & P \ 2 X S 8 ? Q . $ W \ @ ; 1 ; 4 1 \ H p  S , @ J ~ e ! b ! < : K / @ e ( 1 3 J l N k D y > � K = 8  F | _ G = ' \ (    �  k  ~  A    (  �  	  M  �  j  0  Y  �  L  E  �    g  �  �  �  �  e  W  -  �  l  4  �  �  �  p  �  O  �  �  �  \  �  w  �  �  �  l  �  �    
  �    A  
  �  �  �    �  �  e  m�ixռ�j�t��49X%   �ě�;D��<��
;��
<�o=��P<49X<D��<��<���=���=T��=�w=P�`=@�<�h=o=Y�=\)=@�=��m=��=\)=D��=y�#=D��=e`B=�O�=�"�=,1=�;d>V=u=H�9=8Q�=�%=8Q�=�C�=��=��-=�j=aG�=ȴ9=u=�
==��P=�v�>�C�=��=���=���>E��=���=���>1'>$�/B<B�B�Bi�B��B�B
q�B!�B .�BbSB$|�B3 �BΦB5B�lB\�B��BQ�B&2B w�B#��B#C�B1~B#B��B�nBIB��B(Ba�BB�BȢB jKB�FB$�B��Bd�B�(B�MBOqB�CBrBq)B��B��B":�B~cB�B/,B>�A��uB	�B�Bo�B,�BB/��B�B��BELBm:B�B?�B�~B�gBZ�B�B2�B
JSB!=�B HoBA4B$P�B2ȝB��B�B��B��B��B@�B&>{B @�B$/�B#BEB?�B"ÂB��B/�BC�BA4B�BF�B��B��B �\B6�B$�cB��B_�B�5B	(�By/B��BA�BܲBOgBmB"B�B�hB@B?�BA�l�Bh@B?BF�B,�eB/�nB1�B:�B?�B@%B�MC�ZA R$@�l�A��;Ahv2A<�A�:�?�ѓA�C�geA1��AUR6A~"�A�G@z�JA�^A<�??���@�;A%�@��`@�@��@��A���A���A��C��A���A��ZA�}�AF�AW�qA��XAAsA�j>A��A��A�4�A��A��sA5��A���A�hUA�apA��?�\�@r�Ae�B�A�@���C��zBN�AA-�@�+nA���B �C���C�TC�W�@�@�(A�z0Ag�A��A�}�?�A���C�hbA2�wATA~��A�}�@y�IAȀ�A<�?��C@�
.A&�>@��@���@��@�2A���A�~�A�g�C���A��A�E�A��AF�9AU�A�P'A
+]A߀�A�w�A�k=A�� A�8A�/(A63A�s�A��A��A�0�?�S@�Ah�B<�A�g�@��uC��:BcyA� Ag\@���A��B ��C��4C��                     
            G               k   %                            ^   ,                  %   J   	   J   �         	               $   1      3   	   /          �         	   x   	         '                                 '               1         !                     1   )                  %   /         1   %                     -         %      '                     +                                                            )                              /                        +                                 #               '                                 N���N2G�N)Y�N(�O(�=N��N��NC78N0'�N}L\O�qO �N (Nt��N;<2P2OKw�N;9�O��N�� N�#N\��O��NVABOРPu;�O�O�NTOO}�O�)N��O9;O���P!�OQ�O��3O��O���O�%N�2�O���N� �O~�Ov�O�IOp�N7��O�w�N�CP��N�:rO�xO$s8N�dN~oLN^��Ol��N�y�N�w}N'|�O{_  �  �  [  �  �  �  s  �    �  �  �  �  �    
4  �  �  *  �  W  (  a  �  v  
�  P        �  �  �  �  d  �    �  �  #  �  �  �  �  	    �  �  �  �  	  �  r  �  I  D  6  �  �    	轓t���`B���
�u�49X�t��ě�<49X��o;D��<�9X;�`B<#�
<�t�<e`B=@�<��
<ě�<�j<�<ě�<���=C�<�h<��<��=0 �<�=o=\)=��=t�=0 �=49X=C�=0 �==�w=t�=�P=,1='�='�=@�=D��=ix�=8Q�=m�h=T��=q��=y�#=�%>#�
=�C�=�t�=�O�>o=���=�9X=��m>%!#,/0<>C></#!!!!!!��������������������w}����������wwwwwwww��������������������)5BENHB<5)MKNO[\g[UOMMMMMMMMMMroqt���������trrrrrr������������������������������������������������������������������
#(3530#
���bbdhhru���������xuhbABCO[d^[OBAAAAAAAAAA525<HIUTH<5555555555WT[_hnonh[WWWWWWWWWW��������

���������������������������ghipt�����thgggggggg#0<IUZZUOI<0#��������������������������



������������������������X[[^ht�������utsh`[X��������������������������	���������5N[dmplaNB5��������
#%(&
�����zwwvz}�������zzzzzzz�����������������")-3AN[gswvkg[NB>52"��������������������NKKOO[hu�����th[UON��������������������������);BEB8)����#'.340+%#
�����������������������)5>CDB>5)
����#.565-)����IEELN[gtv{zvtkgb[PNI96:<>HU[`^UQH<999999��������������������$")*5@BNTVNB5)$$$$$$�
#/<N^fea^UH<,
���)5BFROEB5)�wuz����������������w��������������������	)-))	6BOZ[WPH>6)	

!#&&''#



	
)5BNUWUPD:5)
_UU]ajmz�����zzmga__')6BO[bb`[NB6)��������

����
##+/-+#
��������������������"*69<6*����������������������
 "
	�������	
##)&#!
�����������������������������


 ����D�EEEEEEEEED�D�D�D�D�D�D�D�D�D������ڼּʼļ������ʼּ������㻷�û˻лܻ޻ܻлû��������������������������������������������������������������T�`�d�l�m�q�s�s�m�m�`�T�M�G�C�B�F�G�J�T�y�������������{�y�x�y�y�y�y�y�y�y�y�y�y�����������������������������������������'�3�5�;�7�3�'�&�����'�'�'�'�'�'�'�'²½¿������¿²¦¦©²²²²²²²²E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͽ�����(�4�:�;�9�4������Խ˽ѽ޽���ʾ׾��������������޾׾Ҿʾɾƾʿݿ������ݿܿڿܿݿݿݿݿݿݿݿݿݿ�������
������������������������������-�:�D�F�P�F�:�-�"�%�-�-�-�-�-�-�-�-�-�-àìù������ÿêÓÇ�z�a�S�J�G�U�a�nÇà�4�A�M�Z�_�c�k�s�u�s�l�f�Z�M�A�3�%��)�4�L�P�Y�e�h�e�e�Y�L�J�D�K�L�L�L�L�L�L�L�L�������������������p�f�_�W�M�L�Y�f�r����Ľн׽ݽ�ݽսнĽ��������������������S�_�j�l�x�~�z�x�u�l�i�b�_�S�P�K�O�N�S�S���������û̻Ļû��������������������������������ɼȼ������������������������������������������������������������������������������������������������������#�I�X�m�x�y�v�n�b�U�0������������ؿ������ �&�(� �������ܿοɿпݿ�F$F1F=FJFPFJF=F7F1F$FFFF!F$F$F$F$F$F$������������ �%�������������żŻſ�ƿ�����(�5�7�-����ݿۿؿؿڿݿ����6�B�B�B�<�<�6�)�&�'�)�4�6�6�6�6�6�6�6�6�����������������������u�s�f�f�m�s�}�����"�)�*�'�&�"��	����׾Ǿ����þ׾����"�/�;�G�K�M�J�9�"��	�������������	������!�.�2�:�K�:�.�!�����������āčĚĦĳĿ��������ĿĳĦč�t�h�a�h�oā���)�6�B�H�K�K�J�B�6�)�����������¿������������¿²¦¡¦ª²¿�B�N�[�g�k�h�e�\�[�N�B�5�1�)�'�)�.�5�8�B����"�)�4�4�)������� �������A�N�Z�n�y�{�s�k�Z�N�5�(������(�5�A��(�2�4�8�4�3�*�(�������������"�'�/�2�0�+�"��	�������������������������������������������������������������������������	������������������Óàìù��������ùóìàÓÇÁÀÇÎÓÓ��'�*�3�4�=�3�'�'������������~�����������ź����������~�r�i�e�j�r�t�~�m�m�o�m�`�]�T�G�;�.�"�.�9�;�G�T�`�j�m�mƳ��������������������ƧƎ�u�h�[�W�sƎƳ�{ŇŔŠťŭůŲŭŤŠŜŔŇŇ�|�{�z�{�{�������4�;�H�M�J�A�4�'����������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ǔǡǭǲǴǭǭǡǔǈ�{�q�z�{ǈǐǔǔǔǔ�y�����������������{�y�w�s�x�y�y�y�y�y�y�S�`�l�n�l�k�c�`�S�J�G�O�S�S�S�S�S�S�S�S�����ûлһܻ���ֻܻлû��������������n�t�{�{��{�n�b�U�L�U�U�b�i�n�n�n�n�n�n�6�C�O�R�X�O�O�C�6�,�*�#�*�+�6�6�6�6�6�6E7ECEPEZE\EbE\EPECE?E7E3E7E7E7E7E7E7E7E7EiEuE�E�E�E�E�E�E�E�E�E�E�E�E�EuErElEiEi # P b 2 P S 8 > Q 0  W \ / ; I < : , T H q  S - = D ~ e  a   5 6 K -  G ( 1 3 J l G b D y : � K = 1  F r G & = ' \ #    �  k  n  A  �  (  �  k  M  �  w  0  Y  t  L  �  �  ^    �  �  �  (  e  =    e  l  4  n  Q  �  a  �  O  f  +  '  \  �  J  �  �    v    �  M  
  �      [  �  �  t  �  �  �  e  K  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  �  �  �  �  �  �  �  �  e  C    �  �    ?  �  �  n  #  �  �  �  �  �  �  �  �  �  �  �  �  c  N  A  4  &    
  �  �  R  X  ]  c  n  {  �  �  �  �  r  6    �  �  �  j  <    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    ^  <    �  �  �  �  �  �  �  �  �  �  �  �  �  m  S  2    �  �  �  �  �  �  �  �  �  �  �  �  �  x  o  g  ^  Y  ]  `  d  g  k  o  s  l  f  ^  V  L  A  3  $      �  �  �  �  �  �  �  �  �  r  �  �  �  �  |  `  [  p  �  �  �  �  �  i  A    �  �  =        #  !                 �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  \  =     �  �  �  T    �  �  �     X  y  �  �  �  ~  k  J    �  �  h    �    :  F  �  �  �  �  �  |  l  ]  M  =  -    	   �   �   �   �   �   �   �   �  �  |  x  t  p  l  h  d  _  [  V  N  G  @  8  1  *  #      w  �  �  �  �  �  �  �  �  �  �  �  �  �  o  W  ;    �  �          
       �  �  �  �  �  �  �  �  �  �  �  �  �    �  	Z  	�  	�  
%  
2  
  	�  	�  	�  	L  	  �  �  :  d  *  �  �  �  �  �  �  �  �  �  �  i  K  .  
  �  �  R  �  �    �  �  �    4  c  �  �  �  �  �  �  �  �  �  P    �  O  �  z  
    #  *  '      �  �  �  �  {  T  4    �  �  �  �  O    �  m  q  u  z  �  �  �  x  l  Z  @    �  �  >  �  .  �   �  W  Q  K  F  @  =  :  7  7  :  >  A  B  @  >  =  S  o  �  �  %  &  '  &        �  �  �  c  /  �  �  q  D     �   �   �  �    6  N  \  a  ^  Y  P  G  9  %    �  �  �  ^    �  S  �  �  �  �  �  �  �  �  �  �  �  �  x  S    �  p  =     �  o  u  s  k  \  G  .    �  �  �  |  S  X  7  �  �  0  �  `  
�  
�  
�  
�  
�  
r  
=  
   	�  	\  �  �    �  %  �  >  `  �  B  �  �    #  ;  K  O  F  -    �  �  [    �  b  �  a  �         �  �  �  �  �  v  _  I  3    �  �  �  �  z  |  ~  �       �  �  �  �  b  @  3  =  G  A  )    �  �  �  �  \  =  �      �  �  �  �  �  �  �  i  G    �  �  M  �  �  8  �  F  Z  l  |  �  �  �  �    9  Y  v  �  �  �  �  	  i  �  a  x  �  �  �    w  k  Z  D  &    �  �  �  G  �  �  )  �    �  �  �  �  �  �  �  �  �  �  y  W  *  �  �  9  �  s  %    �  �  �  �  �  �  �  �  �  �  �  x  F    �  R  �    4  >  d  ]  U  O  K  F  ?  8  ,         �  �  �  �  �  �  �  x  �  �  �  �  �  �  l  /  �      
�  
+  	�  	  K  S    ^  �  T    �  �  a  �  m  �      �  �  V  �     B  �  H  	)    u  �  �  �  �  �  �  �  `  8    �  �  �  ^  +  �  w    �  �  �  �  �  ~  j  T  @  )    �  �  �  �  �  _  =        #        �  �  �  �  �  �  �  �  m  X  C  '    �  �  S  �  �  �  �  �  �  �  �  �  �  q  H    �  �  I  �  �  )  �  �  �  �  �  �  �  �  �  �  �  �  v  g  X  I  :  +         �  �  �  w  Z  :    �  �  �  r  @    �  i    �  B  �  �  �  �  �  �  �  �  �  �  �  �  a  .  �  �  x    �  M  �  �  �  �  �      �  �  �  �  �  R    �  u  ;  �  �  p  1  �  �    2  \  t  ~  |  l  J    �  �  G  �  K  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  b  
  �  �  o  ?    �  �  m  .  [  �  �  �  �  �  v  V  )  �  �  �  p  .  �  K  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  8  �  �  �  s  `  L  �  �  �  �  �  y  E    �  �  i  ,  �  �  M  �  �  G      	  �  �  �  �  �  i  J  3      �  �  �  m  E    �  �  �  �  �  �  �  �  t  \  A    �  �  �  >  �  �  |  N  	  �  �  �  �  �  �     �  	  G  j  j  0  �  �  #    �  s    �  	8  �  �  �  �  �  }  k  V  =    �  �  �  w  ;  �  �  E  �  �  �  �  �  �  �  :  H  H  A  6  &    �  �  �  �  P  �  �  P    +  =  9  (    �  �  �  �  �  �  t  X  ;  !    �  �  �  	j  
�  r  �  r  �    #  4  3    �  r  �  @  
o  	H  �    8  �  o  Z  G  6  %      �  �  �  �  �  �  �  k  O  5      �  m  X  E  +    �  �  �  a  !  �  �  7  �  �  .  �  d   �    �  �  �  �  �  �  �  _  *  �  �  y  :  �  �  p  +  �  �  	�  	�  	�  	�  	�  	�  	k  	0  �  �  h    �  M  �  v  �  R  �   