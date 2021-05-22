CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�G�z�H     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�d�     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��P   max       <T��     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @F��
=p�     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @v������     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @R@           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @���         $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �'�   max       <o     $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B5:$     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��'   max       B4�_     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >2�   max       C��     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >B�   max       C��     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�}5     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Fs�   max       ?��;dZ�     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��P   max       <D��     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @F��
=p�     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @v������     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @R@           �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�I�         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         G   max         G     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?��;dZ�     �  _�      
      �                        4      &         4                  )               #      _               n      	                           	         
                  &   "                           
   '                  '   N���N�ˈN�20P�d�O���OM��O�@Nd�N�=�O�J�O7�BP��N3�O��Np�#N�*�P�}5Ng#^O�ͰOZYJN�"UO"VP((�N���O��N�v�ON��P$��O&UPAC N��^N²�O>uNXX�PV�O
�O#y7N`�O��XO8'�N�N4�jN=oNA4�NӉ�O2�Nc�O�6�N���NWR�N��O�"�Nf��N�Y�O��O���O)�Oc�N�njO���O��'N�KmNӼ�M���O�O4~\N+'�Ow�OW��N�
qO"�O�� N��+<T��;�`B;��
%   �ě��t��t��t��#�
�#�
�T���u��C���t����㼛�㼛�㼣�
��1��1��9X�ě����ͼ�����h��h���������o�o�+�C���P������w�#�
�'',1�0 Ž0 Ž49X�8Q�<j�@��]/�]/�]/�]/�ixսq���q���}�}󶽁%��+��C���C���O߽�O߽�O߽�hs��hs��hs���T���T��{��"ѽ�"Ѿ�P����������������������������������������)36>76-)
)5BN_ksyugNB5.
)06B[hpsv�~xth[XB6+,0FHUanqssz�znaUHACCF����������������������������������������46:BOPT[\a[OKBB64444�����#.0:90#
�����
#/<DHPSI<8/#
��������������������!#/;<<</###!!!!!!!!�����������������������|��������������� �����#U{�����{bN<0
����������������������HT_ajmosvuoiaTHC=<>H����������������������������������������%/<HUaggda\UH<:/-)%%)5B_g������t[B5&! #/<HHUSHD</'#��)68BEB)���ptw������������tpppp!"&)/;;BAABCB?;3/-#!oy�������������ztnmowz���������������wUZar�����������zmULU�������� �����������W[_hit������thb[WWWW5;HJTamiaXTKHC;:3535Uailmnoma`]XUUUUUUUU#/HTnwtzygQH</##���������������������������������������������������������vz|��������������wv)*6BOSW[XOB6)'`hnt������tnh``````��� ����������������������������������������������������������� )1-)����]gqt����������}tga_]��������������������m{���������{qjhjheem����������������������������������������[[hot|th[V[[[[[[[[[[������������������MU]aefgdbaUTRQMMMMMM������������������������� 	�������&)1BN[gt~tog[NB5)""&���������������������� 
!
��������������������������P[gt�������xwtg^XNIP_mt�����������tic_\_���
#),#
��������3<HUXaabcaUHB<;73333]anoonna_\]]]]]]]]]]#0<MUVWUPNLID;2#,0;<INPQPNI<40+)'(),������������������������	
�����������!*,,)#
������gghty����������tjgggZ[\_bgmt��������tg[Z���������������������������������������ؾ(�#�"�(�+�4�A�M�O�U�M�I�A�4�(�(�(�(�(�(��������(�4�?�A�J�A�7�4�(�����ֺкɺƺźɺֺ̺���������������ֺֿ�������5�N���������������������s�N���;�3�,�&�,�;�G�`�y����������������m�G�;�H�E�@�@�D�H�T�a�q�z�����������w�m�a�T�H������������������������������������������������������������������������������z�|������������������������������ܻл̻ʻƻ����ûǻлܻ����
�������ììáÞÛÝàæù����������������ýùì�4�/�/�5�A�Z�g�s�{���������������g�N�A�4�t�r�m�t�y�t�t�t�t�t�t�t�t���
�	�
���4�@�M�Y�f�t�|�r�M�4�'�����ݻ����������������������	� �	����!�"�'�/�9�1�/�*�'�"�������r�J�7�D�s���������������������������.�&�)�.�;�;�G�T�Y�T�S�G�;�8�.�.�.�.�.�.ď�t�h�c�h�l�tāčĚĳĿ����������Ŀĳď�x�l�_�U�K�F�:�!��	���-�S�_�g�l�o�t�x�������������������������������������������������������������������������������˾ʾľ����׾߾���	��.�4�3�'����	����)�'�(�&�)�)�6�B�K�O�Z�T�O�L�B�6�)�)�)�)�����������������������������������������������s�r�i�g�d�g�s�}���������������������������s�g�d�s�������������������������h�C�1�/�2�F�O�hƀƎƬƳ����ƾƺƳƚƁ�h���������������������$�&�&�$�����e�@�"���������'�L�Y�m�u�y�r�r���~�e�"����	�����	��"�.�.�;�>�G�L�G�;�.�"���������������������������������������������	����5�A�G�N�R�N�C�A�5�(���"��"�/�2�;�H�P�I�H�;�/�"�"�"�"�"�"�"�"E�E�E�E�E�E�E�E�E�FF1FBFUFGF+FE�E�E�E��ѿϿÿ������������Ŀݿ��� ������ݿѿT�G�D�=�:�:�:�;�G�T�Y�`�y�}�������y�`�T�����������������������������������������U�L�I�?�8�:�@�N�b�n�{ŇņłŁ�{�w�n�b�U�l�b�_�S�S�J�J�S�_�l�x��������������x�l�����������������������������������������;�;�/�;�H�P�T�a�l�a�T�H�;�;�;�;�;�;�;�;ùôìììöù������������ÿùùùùùù����������������������������������������ŠşŚŔŒōŊŋŔŠŤŬŭųŭŭũţŠŠ�m�i�m�o�t�u�t�z���������������������z�mF1F.F.F1F9F=FJFUFTFJFCF=F1F1F1F1F1F1F1F1�-�-�4�J�_�x�����ûϻֻ׻лû������l�:�-ŹŮűŹ����������������������������ŹŹ��������������"������������������ùɹι̹ù��������������������Y�L�B�>�A�L�Y�r�������������}�x�x�r�e�Y�׾ѾʾǾʾ׾������������׾׾׾׾׾��)�#�������%�)�.�6�?�B�C�H�B�6�)�)��ìàÑÒàìù�������'� ����������=�9�0�������"�0�;�T�c�g�g�a�V�I�=����������$�+�0�3�1�0�-�$�#��D�D�D�D�D�D�D�D�EEEEE!EEEE	ED�D�4�1�4�?�@�M�Y�_�f�r�z�s�r�f�\�Y�M�@�4�4��������������)�5�A�N�[�d�b�[�N�F�5��u�t�w¦²¿����������¿¦���ݿԿѿ׿ݿ�������������������������������&�"�������������ѿοοѿݿ����ݿѿѿѿѿѿѿѿѿѿѽ��������������������Ľнݽ�ݽнĽ���������������(�4�A�N�\�Z�X�M�A�4�(���������ʼϼּ������ּʼ������������.�&�!���!�.�:�G�L�S�`�f�`�S�Q�G�C�:�.�y�p�t�~�����������Ľͽ׽нĽ����������y�H�C�<�1�/�-�.�/�<�H�U�a�h�a�]�X�U�N�H�HĚĖčā�t�o�h�d�`�f�h�tāčĖĚĠĞěĚĚĖĠĤĤġĠĦĳĿ������������ĿĳĦĚ���������������������������������������� E 3 < S 3 9 O A j V U < G , ) _ T ^ e k = 8 a B e l � : : E h . 3 t B |  a > * � y _ V � ` Y r M d Q T \ l T W P / } ?  = A = U S | 8 < + 0 V 4    �  �    F    �  �  K  �  ~  �  l  W  �  q  =  ~  �  �  6  �  w  `  6  �  �  �  �  x  �  )  �  �  �  �  �    �  ?  �  �  w  |  {  l  �  �  O  -  �  M  T  �  =  k  u  �  G  �  �  j  �    "  �  �  �  9  �    s  �  �<o�D���D���
=q�ě���1��/�49X�T����h��/��+���aG���h�ě���\)�����D���t���/�P�`��+�0 Ž�P�o�'���0 ž%��w�H�9�49X��w���8Q�@��0 Žy�#��7L�0 Ž8Q�Y��@��P�`�Y��]/�����%�ixս�����w�y�#��t��ě��\������vɽ�hs��E��Ƨ������㽓t����T��;d���P��9X��
=����   �z�'�B��B!_�B�B�aB�B7�B��B!��B��B$B+VB/�B#FB��B�B�>B&�B1
A���B��B��B3�B	[ByB��B��A�#�B�B �_B}�BBB4�A��AA���BC"B*u"B,�B�5BL�BݝB_�A��B�gB5:$B��B
M�B��B(��BhB^�B�tB"L�B
8B�B��B�B�B��B*��B	q�B
S�B8BZ�Bn�B&�B&:�B-XB"gB?�B
F�B	ߞB10B]�B�=B!=B�WB��B@�BAB��B!��B�XB#��B�B=�BˌBFCB9HB�	B&�B?�A��B>%BBAKB	�TB�_B��BJ�A��B\�B αBI�B=?B@�A�[�A�y�B=DB*4	B,; B9IBY�B��B��A��'B�qB4�_B:YB
llBB);�B�B�0B��B"RdB>�B��B�B�6BB(B�|B*��B	�2B
}WBʐB��BGQB& �B&@�B-K9BC�B7�B
�\B	�5BY>B��A9�jA6��@B��A���Aj A�A��&ALi�AHk�@�o�A�c�A���A�{@�?@�r�A�$�A��Ac�xA�mj@c�A��A���AY�A�˱A�M�A�0A��{B�@B��?�i�A_Z�A"0�A�o�A��C�{�A|;@Ai�A�[<A�}�@�_X@�ןA��A�QlAL�A��UA�E�C��@�3�A���A�$�>2�?�j�AU
�AּNA�@�B
�2B	'HC�K�@�1A�'�A���A~�)A��A|�`A%^TA7�MA ��A�4A!��A�j�AݻYA�LQB�A:y&A6�@Ca[A��1Ag bA��HA�u0AK�AI�@��UA�t�A�yoA��b@���@��A�^ A���Ac!�A�9@s�?A��VA�iA[hAדOA��GA��A��0BBFB�u?�5A`?A"�A��wA�WC��YA~$YAf��A�'�A�2@�e@�.�A��A͖AL��A�A���C��@�*�A�W4A���>B�?�aAT�;A�QA��BnB	D�C�T�@ق�A�}�A�$A~��A�KA|��A#wA8�aA]�AA#�gA�nA���A┼BL�            �                        4      &         4                  *      	         $      `               n      
                           	   	      
                  &   "                              '                  '               ;   !                     #      !         7                  -               )      1               3                                       %                     '                                                   !               %   !                           !         7                  %               %                     3                                       %                     '                                                      N���N�ΌNٸ�P6O���OM��N�L�Nd�N�=�O�J�O�0O��pN3�O��Np�#N�*�P�}5Ng#^O�2�OD�N�"UO"VO�\�N���O��N�v�ON��P �FO&UO���N��^N�e�O'V�NXX�PV�N�,�O#y7N`�O��XOQN�N4�jN=oNA4�N��O2�Nc�O�6�N���NWR�N��O�"�Nf��N�Y�O��Osa�O)�Oc�N�njO���O��N�KmNӼ�M���O�Os�N+'�Ow�OW��NŸZO"�Og�N��+  �  4  �  �  �  �  "  /  �    V  �    �  �  �    o  u  �  h  �  %  A  w  �  :  $  6  	;  �    2  �  �  `    V  X  �  �    �  E    >  �  J  }  (  �  !  �  U  �  �      �    �  �  �  �    �  9  &  �  i  �  P  N<D��;ě�;�o�e`B�ě��t��u�t��#�
�#�
�e`B������C���t����㼛�㼛�㼣�
��9X��9X��9X�ě��+��`B��h��h���t������-�o�+�C��C���P��w����w�#�
�8Q�',1�0 Ž0 Ž8Q�8Q�<j�@��]/�]/�]/�]/�ixսq���q������}󶽁%��+��C���hs��O߽�O߽�O߽�hs������hs���T���T��-��"ѽ�`B��P����������������������������������������)16<65,)	$)5BNW`cb`[NB5) 06B[hpsv�~xth[XB6+,0FHUanqssz�znaUHACCF����������������������������������������46:BOPT[\a[OKBB64444�����#.0:90#
�����	
 /<AHNOHE<:/#	��������������������!#/;<<</###!!!!!!!!�����������������������|��������������� �����#U{�����{bN<0
����������������������EHT`dmrutniaTHD><>BE����������������������������������������%/<HUaggda\UH<:/-)%%.5BNWd�������tN8-)*.#/<CHRPHB</+#��)68BEB)���ptw������������tpppp!"&)/;;BAABCB?;3/-#!z������������zusstzwz���������������wx~���������������vux�������� �����������`hjt������thc]``````6;HT`agda_WTIHF;5646Uailmnoma`]XUUUUUUUU#/HTnwtzygQH</##���������������������������������������������������������vz|��������������wv")-6BKOQTVOKB4)`hnt������tnh``````��� ����������������������������������������������������������)/+)�����]gqt����������}tga_]��������������������m{���������{qjhjheem����������������������������������������[[hot|th[V[[[[[[[[[[������������������MU]aefgdbaUTRQMMMMMM������������������������� 	�������'BN[gmokgd[XNB5/)##'���������������������� 
!
��������������������������P[gt�������xwtg^XNIPcgt�����������tlfb`c���
#),#
��������3<HUXaabcaUHB<;73333]anoonna_\]]]]]]]]]]#0<MUVWUPNLID;2#.07<ILNOOMIG<0-*))+.������������������������	
�����������!*,,)#
������kt~����������tsikkkkZ[\_bgmt��������tg[Z���������������������������������������ؾ(�'�%�(�-�4�A�M�N�R�M�A�A�<�4�-�(�(�(�(�����	���(�4�=�A�E�A�4�4�(�����ֺѺɺƺƺɺͺֺ����������ֺֺֺ��Z�A�:�.�,�4�A�Z�s�������������������g�Z�;�3�,�&�,�;�G�`�y����������������m�G�;�H�E�@�@�D�H�T�a�q�z�����������w�m�a�T�H����������������������������������������������������������������������������������z�|������������������������������ܻл̻ʻƻ����ûǻлܻ����
�������ùöìãàÜÞàêù������������������ù�N�A�8�8�A�Z�g�s�������������������s�g�N�t�r�m�t�y�t�t�t�t�t�t�t�t���
�	�
���4�@�M�Y�f�t�|�r�M�4�'�����ݻ����������������������	� �	����!�"�'�/�9�1�/�*�'�"�������r�J�7�D�s���������������������������.�&�)�.�;�;�G�T�Y�T�S�G�;�8�.�.�.�.�.�.Ěđ�u�h�h�uāčĚĳĿ����������ĿĳĦĚ�_�X�M�F�:�!��
���-�S�[�_�f�k�m�r�l�_�������������������������������������������������������������������������������˾׾˾þɾ׾����	��"�,�0�-���	�����)�(�)�'�)�,�6�B�I�O�W�R�O�J�B�6�)�)�)�)�����������������������������������������������s�r�i�g�d�g�s�}���������������������������s�g�d�s�������������������������O�C�8�4�7�>�O�\�h�uƎƤƻƳƨƚƎ�u�h�O���������������������$�&�&�$�����3�'���� �'�3�@�L�W�Y�b�h�h�e�Y�L�@�3�"����	�����	��"�.�.�;�>�G�L�G�;�.�"������������������������������������������������(�1�5�A�E�N�O�N�?�5�(���"��"�/�2�;�H�P�I�H�;�/�"�"�"�"�"�"�"�"E�E�E�E�E�E�E�E�E�FF1FBFUFGF+FE�E�E�E��ѿпƿ��������Ŀ˿ѿݿ�����������ݿѿT�G�D�=�:�:�:�;�G�T�Y�`�y�}�������y�`�T�����������������������������������������U�L�I�?�8�:�@�N�b�n�{ŇņłŁ�{�w�n�b�U�l�g�_�W�S�L�M�S�Y�_�l�x�����������z�x�l�����������������������������������������;�;�/�;�H�P�T�a�l�a�T�H�;�;�;�;�;�;�;�;ùôìììöù������������ÿùùùùùù����������������������������������������ŠŝŔŔŏŌŎŔŠŢũŭůŭŪťŠŠŠŠ�m�i�m�o�t�u�t�z���������������������z�mF1F.F.F1F9F=FJFUFTFJFCF=F1F1F1F1F1F1F1F1�-�-�4�J�_�x�����ûϻֻ׻лû������l�:�-ŹŮűŹ����������������������������ŹŹ��������������"������������������ùɹι̹ù��������������������Y�L�B�>�A�L�Y�r�������������}�x�x�r�e�Y�׾ѾʾǾʾ׾������������׾׾׾׾׾��)�#�������%�)�.�6�?�B�C�H�B�6�)�)��ìàÑÒàìù�������'� ����������=�!�����$�0�5�=�Q�V�a�e�e�b�^�V�I�=����������$�+�0�3�1�0�-�$�#��D�D�D�D�D�D�D�D�EEEEE!EEEE	ED�D�4�1�4�?�@�M�Y�_�f�r�z�s�r�f�\�Y�M�@�4�4��������������)�5�A�N�[�d�b�[�N�F�5��~�{�¦²¿����������¿¦���ݿԿѿ׿ݿ�������������������������������&�"�������������ѿοοѿݿ����ݿѿѿѿѿѿѿѿѿѿѽ��������������������Ľнݽ�ݽнĽ��������������(�4�A�K�M�Z�T�M�A�4�(���������ʼϼּ������ּʼ������������.�&�!���!�.�:�G�L�S�`�f�`�S�Q�G�C�:�.�y�p�t�~�����������Ľͽ׽нĽ����������y�<�4�/�/�/�/�<�H�U�a�c�a�[�U�U�H�<�<�<�<ĚĖčā�t�o�h�d�`�f�h�tāčĖĚĠĞěĚĳĨħģĢĦĮĳĿ������������������Ŀĳ���������������������������������������� 6 3 : A 3 9 U A j V M D G , ) _ T ^ b j = 8 ] = e l � 1 :  h " 2 t B j  a > * � y _ V | ` Y r M d Q T \ l T O P / } ?  = A = U K | 8 < . 0 / 4    �  �  �  w    �  �  K  �  ~  z  �  W  �  q  =  ~  �  k    �  w  �    �  �  �  F  x  �  )  �  q  �  �       �  ?  L  �  w  |  {  $  �  �  O  -  �  M  T  �  =  k  4  �  G  �  �  ,  �    "  �  S  �  9  �  �  s  �  �  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  -  1  4  1  ,         �  �  �  �  �  j  D    �  �  �  �  �  �  �  �  �  �  �  y  b  G  &  �  �  �  i  C  !  �  �  �  O  	0  	�  
j  
�  F  �  �  �  �  }  ?  
�  
p  	�  	M  l  <  �  �  �  �  �  �  �  �  v  e  N  5  )  #    �  �  �  s  A   �   �  �  �  �  �  �  �  �  �  �  �  m  T  :      �  �  y  ;   �  �  �  �  �         !        �  �  �  �  o  C    �  �  /  -  ,  *  )  '  %  $  "  !       �   �   �   �   �   �   �   �  �  �  �  �  �  �  }  t  k  a  W  N  D  7  #     �   �   �   �        �  �  �  �  �  u  D    �    �  �  �  x  9  �  u  S  U  P  C  .    �  �  �  �  �  �  �  z  y  �  s  M  #  �  l  �  �  �  �  �  �  �  �  s  @    �  �  p  /  �  "    �    C  S  N  D  :  1  '      
     �  �  �  �  �  �  A  }  �  �  g  I  /          �  �  �  �  N    �  h    �  �  �  �  �  �  �  �  t  d  R  =    �  �  �  G  	  �  �  A   �  �  �  �  �  �  �  �  �  �  r  d  V  R  T  W  Z  k  �  �  �    �  �  z  B    �  �  ~  N    �  �  ~  I    �  8  �  	  o  m  k  i  _  T  I  J  P  V  X  S  O  I  ?  6  ,  $      t  p  \  C  (    �  �  �  U    �  y    �  T  Q  �  �  !  �  �  �  �  �  �  �  }  Z  5  #  �  �  �  |  S  ;    �  �  h  d  `  ]  X  R  L  F  =  1  %      �  �  �  �  �  �  �  �  �  �  �  �  z  l  w  k  b  Z  J  ,  �  u    �  ?  �  L  �  �  �    #    �  �  �  �  �  �  �  �  w  9  �  	  /   �  8  >  @  =  2    �  �  �  �  ^     �  �  n  @  1  @  R  a  w  t  q  m  g  _  O  ?  +    �  �  �  �  �  R     �   �   Q  �  �  �  �  �  �  �  x  l  `  T  H  <  1  (      
     �  :  /  "    �  �  �  �  �  �  �  �  �  �  �  Q    �  �  o         $      �  �  �  �  `  5    �  �  6  �  �  A  y  6  -  &  #  !            �  �  �  �  �  �  �  u    �  K  �  �  �  �  �  �  	
  	,  	:  	4  	  �  C  �    8  G  V  �  �  �  �  �  �  �  �  �  �  �  z  e  L  4    �  �  �  \  /  �        �  �  �  �  �  �  ~  P    �  �  @  �  �  l  1  0  1  .      �  �  �  �  �  �  d  E  !  �  �  �  z  c  T  �  �  �  �  �  �  �  �  �  �  �  j  Q  8      �  �  �  �  �  N  "    �  �  �  6  �  �    
O  	|  �  �  �    R  A  �  Z  \  _  \  L  =  .  !    �  �  �  �  �  �  �  y  ]  >      �  �  �  �  �  �  �  {  f  L  6  #    �  �  �  �  �  R  V  G  9  +            	     �  �  �  �  �  �  �  �  r  X  F  7  3  %  	  �  �  �  l  >    �  �  R  �  �  $  �  �  �  �  �  �  �  �  �  �  �  c  ;    �  ~  6  �  �  N  a  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  z  e  ?    �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  ^  ?    �  �  �  M    �  E  9  -  !    	  �  �  �  �  �  �  �  �  �  �    l  X  E            
    �  �  �  �  �  �  �  �  �  �  �  �  j  >  8  1  (        �  �  �  �  �  �  �  |  Y  1        �  �  �  {  u  n  c  X  E  1      �  �  �  �  �  �    Q  J  ?  '    �  �  �  �  �  �  �    Q  "  �  �  v    h   �  }  s  h  Z  L  <  +      �  �  �  �  �  �  �  y  b  ^  [  (  &  $  "              #  &  )  ,  ,  *  '  %  "       �  �  �  �  �  �  _  =    �  �  �  �  q  N    �  �  >  �  !          �  �  �  �  b  :    �  �  �  b    �  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e    �  q     U  5    �  �  �  �  a  s  �  m  I  %    �  s    �  �  �  �  �  `  ,  �  �  �  R    �  e  (    �  �  T  �  M  �    x  �  �  �  �  m  6  �  �  u  (  �  �  #  �  V  �  �  M  �        �  �  �  �  �  �  q  ]  <    �  �  [  �  D  �   �       �  �  �  �  �  {  P     �  �  k  (  �  �  o  -  �  :  �  �  �  �  �  �  o  ]  K  :  )      �  �  �  �  w  K        �  �  �  �  �  �  �  �  z  `  D  &    �  �  1  �  �  �  �  �  �  �  �  t  Y  :    �  �  d    �  g    �  d  �  �  }  n  _  P  A  0      �  �  �  �  �  �  n  \  J  8  %  �  �  �  �  �  �  �  �  �  v  l  `  M  :    �  �  q  0   �  �  �  �  �  �  �  �  �  �  �      (  0  -  *  '  $  "        �  �  �  �  �  �  �  �    c  G  )    �  �  �  A    �  �  �  �  �  �  �  �  i  :  �  �  ,  �    d  �  �     �  9  3  -  (  "            	      �  �  �  �  �  �  �  &      �  �  �  �  �  �  �  u  a  Q  @  4  -  '  *  1  8  �  �  �  �  k  H  $     �  �  �  z  L    �  �  W    �  u  b  g  i  d  V  B  ,    �  �  �  }  L    �  �  �  ]  _  %  �  t  h  Z  K  ;  +    �  �  �  M    �  e    �  �  �  �  �  �  H  H  0    �  �  |  <  �  �  T  �  �  &  �  /  �  <  N  .    �  �  
�  �  �  5  �  �  ,  �  �  )  �  �  o  ,  �