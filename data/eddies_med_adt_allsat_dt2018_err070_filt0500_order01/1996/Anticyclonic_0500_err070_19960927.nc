CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?��\(�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�1�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��+   max       =��      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @F�33333     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @vpQ��     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�@          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �L��   max       >�=q      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B2�9      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�i�   max       B2��      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?���   max       C�g�      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C�h�      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       Px��      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Y��|��   max       ?�-      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��+   max       >#�
      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @F�33333     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�H    max       @vpQ��     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�*@          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�g�   max       ?�L/�{J     �  T                              F         %      l                      c   +   J      �   "            $            L                  6                     2      #   *         	   	   y   �         )N�0xNAJ�Osk
N�H�N��N
�N�nO,��Nq�P�7OA�+N�@O}�qM��P�1�NЋSO�]�Oy��O �oN`+�O%J�P�/PxO��=O�*VP|��O�%N�z�N3U^Ou�JPa0N$��NTG{N��[PO�;N� �N�3CPS�Nȵ�N0̂Ojc�O��0O�˹O��RO	O�oO\3�O���O�O���Pi(N��N�׭N�AN�.�P��O�.yN�U�N��;O4ᙽ�+��`B�T���D���#�
��`B��o;o;D��;�`B<o<T��<e`B<u<�o<�t�<�1<�9X<�/<�/<�`B<�`B<�`B<�<�<�<�<�<�<��<��<��=o=+=C�=��=�w=�w='�=,1=0 �=49X=49X=49X=49X=<j=<j=H�9=ix�=u=y�#=�7L=�hs=�hs=��P=���=���=�{=��=��#/<<B@<1/$#�����������������������)5BWRK=5)��BBEOX[dga[OHEBBBBBBBqnnt������������tqq$)16BCEB964)$$$$$$$$��������

�����������������������������������������������������#08:<<60#
�����������uh\YV\dhu���//2<HPU\abaURH<80///��������������������[W[hhhlmihe[[[[[[[[[�����!#.)�������^abhjt��������th^^^^#0<IT[ZPI<0#��������������������XY[[ht��������|tnhaX�������� �����������������	���������)5N[gptoeNB���������
/42+
��������������������������������������������)Nguv��g[TL5���")3BN[gtxzzyrg[NB5��������������������������������������KHHJO[ht������yth[OK{}�����������������{��������� ����������\`gmt{}���xtig\\\\\\
 !#%&*..(#"

������)BLL>6)����B=?>HUZ_[UNHBBBBBBBB"$)458BNSUNB953)""""be��������������tib)26>=6)(#"*66;B6*((((((((((��������������������������������������������)5CNVUMB5)����� +1/5:85)��MKGJN[gituyvtrg[VNMMzwvy���������������z 
#/<HU]aUH</)#
	B[^hphf[B6)T[agmz�����}zqmaYUTT")6BN[cda[VOB))5BNSTUQNB5) #+///.,#

###��������������������*+6:>:6*'�
"!
w||���������������zw��������
 
����
"#'(%#
�����������������������������

�����D�EEEEEEED�D�D�D�D�D�D�D�D�D�D�D������޼ּ˼ʼ��ʼּټ��������T�b�m�p�v�z�x�r�t�m�`�T�G�C�A�C�G�N�M�T�����������������y�w�p�y��������������������������������������������������������ÇÍÓÖÔÓÇÄ�z�o�zÄÇÇÇÇÇÇÇÇ¿��������¿²ª§¦¤ ¦µ¿���'�5�@�D�L�S�L�K�@�3�'�������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E;��(�4�?�E�E�@�4�(����ؽȽ����ݽ��	�����޾Ѿʾƾ��������ʾݾ������	�	�������	������������������������������4�M�Z�d�f�m�u�s�m�f�Z�M�A�4�(����(�4�-�:�>�F�F�F�:�-�!��!�,�-�-�-�-�-�-�-�-àù������øøàÇ�a�X�\�B�6�7�D�M�zÓà�@�L�Y�b�e�q�m�f�e�Y�U�L�@�?�<�=�@�@�@�@�Y�d�r���������������������f�]�R�J�H�Y���Ľнݽ����ݽнĽ������|�~�������������������ʼʼƼ��������������������������������$�'�-�'��������������������������������������������������������������#�I�Z�k�x�|�|�x�n�b�I�0����������ؿݿ�����#�-�4�6�>������ѿ������ſ�čĚĦĳĿ��������Ŀĳā�w�h�[�Y�[�hāč������������$�����������ŽŹŶż���O�W�[�^�q�k�[�B�)�'�,�������������O������.�4�:�4�������ݿؿտտֿ߿��B�G�G�J�C�B�A�6�)�#�$�)�*�4�6�<�B�B�B�B�B�M�O�V�[�h�p�h�[�O�K�B�?�@�B�B�B�B�B�B�����������������������s�n�f�c�a�f�v��"�.�6�;�>�<�<�6�.����׾ʾ�����������"�����������������������������������������/�;�?�;�9�/�"������"�$�/�/�/�/�/�/�����!�'�.�:�G�Q�G�:�.�!�������������"�/�;�H�Q�P�C�&��	��������������������)�5�5�)����� ����������(�*�4�5�4�0�+�(������������(�4�Z�f�s������������f�M�4�-�����(��$�'�.�3�9�@�3�,�'�����������y�������������y�p�w�y�y�y�y�y�y�y�y�y�yù������������ìßÓÇÀ�|�zÃÇÕàöù�N�Z�g�r������q�Z�A�5�(������(�5�N����������������������������������������²¿����������¿²¦�|¦²�B�N�[�g�k�j�h�g�\�[�N�B�9�5�+�2�5�@�B�B������������������!������������������"�'�/�1�0�'� ��	�����������������~�����������κ��������������r�c�]�a�i�~ŔŠŧŭűųŭūŠŔœŇł�{�x�w�{ŇœŔ�����)�4�C�N�V�M�D�4�'�����������Ƨ������������� ������ƳƚƏ�w�c�\�hƚƧǭǡǕǔǓǈ�{�p�x�{ǈǎǔǡǭǲǳǭǭǭ�y�����������������|�y�v�x�u�y�y�y�y�y�y�`�l�q�r�m�l�d�`�S�G�E�F�G�N�S�]�`�`�`�`�b�n�{�{�~�{�n�b�U�K�U�Z�b�b�b�b�b�b�b�b�����лջܻ�����ܻû�����������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�DuDkDmDsD{�*�6�C�O�[�\�O�L�C�6�*�'� �&�*�*�*�*�*�*E7ECEPE\E_EcE\EPECE8E7E-E7E7E7E7E7E7E7E7EiEuE�E�E�E�E�E�E�E�E�E�E�E~EuErElEiEeEi % i _ . + ^ | D 0   T 3 ; ] > < - R " u - : I 4 ] E $ v � " ? [ i x 5 6 O H ` e ; D C [ , p k A 9 ' N @ o [ D H  5 N %  �  �  &  �  �  O  A  �  u  P  �  �    (  �  
  B    Z  �  j  8  �  �  Q  3  �    �  �  �  M  �  '  �  �  �  �  *  9  �  x  �  M  &  �      5  �  �    �  �  w  	  �  �  �  ��L�ͼ�9X:�o���
:�o��o;�`B<���<e`B=���<u<��=P�`<���=��=#�
=T��=P�`=H�9<��=<j>%=�hs=���=8Q�>F��=�o=49X=t�=m�h=�7L=\)=�P=�w=�;d=8Q�=,1=�C�=aG�=8Q�=Ƨ�=��=�O�=��=e`B=���=�\)=ȴ9=��=��=��=��=���=��
=��>G�>�=q=���>+>!��B9�B��B��B�uB
v6B�`B��B!%�B�+B$v�B2�9B�B�B��BcGBUqB&B l�BL�B#R8B�xB�iBO�B��B,�Bs6Bf�B4(B�B�<B DWB#XB	�VB$�UB�Bc�BpB�(B̹B05B"�B��B�B�B��B('BM�B��A���BXBhB_zB,�B/�8B'�B�BB1�Bh�B�B>�B�B�gB7 B
N@B��B?�B!>BFuB$XrB2��BrBD�B��B?�BC�B&<]B B%B?�B#;-B�'B}�B2B��B�GB�lBAB~�B@dB��B >[B#?B	��B$�&B7�Bs�BJlB.�B�>B/�B"@wB�2B��B��B	;�B��B�tB@RA�i�B��B¿B?�B,ʁB/��B8{B;7B?6BE�BHwB3�C�U;AN�Ah��AڅA�o}A�{AA��H?�_�C�g�A1��ATc�A�)�A;�`@w��A�%?���@��=A&o2@�;u@���A��hA�A�6A�k�A��aA�
nA���A�i&A�3�AF��AX@j@�x_A��;A�A��A�P�A5��A>t?���Ao,EA�	�A��A�[PA��[A���A�nvA�G@��A�"@���B$�BE3A�0A`A�c@�� C�ޠB �C��C�	�C�V�A �aAh�CA�WA��Aɀ�A���?���C�h�A2�7AS�NA�x�A;@t~�AȀ?נ�@�A&��@���@���A�~A�$A���Aߌ_A���A�z�A���A׈�AځTAF�9AV@��?A�x�A	)�A�87A�/QA5��AB*�?��Ap6�A�u{A�~A�z A�w8A�>�A�~�A��@��A�u�@��B=@BGwA��AeA�D@���C��B �C���C�	}                              F         &      m                      c   ,   K      �   #            %            L                  7                     2      #   +      	   
   	   y   �         )                              %               7      !               3   +   !      9               '            /         +                        '      %         '               )                                                         '                     1                                       )         '                        %               '                           N�0xNAJ�O�xN�H�N��N
�N�UNgػNq�O��:OA�+N��OUjdM��P=�N;�cO���N�y$N�,�N`+�N��Px��O���O�PO�*VO�?�O��wN!�1N3U^O(4O�e�N$��NTG{N��[P@]N� �N�3CP	qrN��N0̂O��Of�ZO�YHO��RO	O�.�O\3�O�{�O�O���Pi(N��Nn@�N�AN�.�O���O!޿N�U�N��;O�   N  �  �  9  �  �  �  �  �  �  -  P  �  �  
d  �    �  �  j  K  
�  h  �    !  v  _  �  �  �  �  �  �  �  1  �  �  W  �  �  �    g  �  �  P  �  D  e  i  �    r  �  [  7  f  �  	Ƚ�+��`B�#�
�D���#�
��`B�D��<#�
;D��<�9X<o<�C�<�t�<u=]/<���<ě�=C�=o<�/<��<��=0 �=t�<�=�G�=C�=\)<�=�P=0 �<��=o=+=<j=��=�w=,1=0 �=,1=u=D��=H�9=49X=49X=H�9=<j=q��=ix�=u=y�#=�7L=�t�=�hs=��P=��m>#�
=�{=��=��##/<<B@<1/$#���������������������)5BGIEB65)�BBEOX[dga[OHEBBBBBBBqnnt������������tqq$)16BCEB964)$$$$$$$$�������
 ��������������������������������������������������������
#*046640#
����������uh\YV\dhu���<326<HKUXUTH?<<<<<<<��������������������[W[hhhlmihe[[[[[[[[[����������������ghjqt|����thgggggggg#0<IRUYYXNI<0#��������������������_^aht��������tjh____�������� ���������������������������)5N[nrmcNA5��������
#'*(���������������������������������������������)5@DED@95)1)*7BN[gqtwvulg[NB51��������������������������������������OLLOP[hq�����}th[WOO����������������������������� ����������\`gmt{}���xtig\\\\\\
 !#%&*..(#"

������)<BDB6)����B=?>HUZ_[UNHBBBBBBBB"$)458BNSUNB953)""""omr���������������to)069:6)(#"*66;B6*((((((((((�������������������������������������������)5<BIOMGB5)����� +1/5:85)��MKGJN[gituyvtrg[VNMMyy|����������������y 
#/<HU]aUH</)#
	)6BNZ[UNF<6)	T[agmz�����}zqmaYUTT")6BN[cda[VOB))5BNSTUQNB5) #+///.,#

###��������������������*+6:>:6*'�
"!
��������������������������� 

�����
"#'(%#
����������������������������	

������D�EEEEEEED�D�D�D�D�D�D�D�D�D�D�D������޼ּ˼ʼ��ʼּټ��������`�h�o�u�t�m�o�m�`�T�Q�G�G�D�F�G�M�T�V�`�����������������y�w�p�y��������������������������������������������������������ÇÍÓÖÔÓÇÄ�z�o�zÄÇÇÇÇÇÇÇÇ¿����¿²°©¦¡¦²·¿¿�'�3�8�>�?�3�'�$�����'�'�'�'�'�'�'�'E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͽ����(�7�;�<�:�4�(�������ӽ˽Ͻ���	�����޾Ѿʾƾ��������ʾݾ������	�	��������
���������������������������޾(�4�A�M�Z�a�d�j�r�f�Z�M�A�4�(�"���"�(�-�:�>�F�F�F�:�-�!��!�,�-�-�-�-�-�-�-�-àìù������ùàÇ�z�a�U�L�L�R�a�zÏØà�L�N�Y�e�h�e�d�Y�L�H�C�K�L�L�L�L�L�L�L�L�Y�f�����������������������f�_�U�O�P�Y���ĽнԽݽ�ݽֽнĽ��������������������������ļż������������������������������������$�'�-�'����������������������������������������������������������������<�U�i�w�{�{�w�b�I�0������������ؿ������"�)�(�#�������ݿӿ˿ӿݿ�ĚĦĳĺĿ��������ĿĳĚĂ�t�h�_�kāčĚ������������$�����������ŽŹŶż�����)�6�B�F�J�L�K�B�6�)������������������*�1�5�-�������ݿۿؿٿܿ���6�B�C�B�>�=�6�)�&�'�)�1�6�6�6�6�6�6�6�6�B�M�O�V�[�h�p�h�[�O�K�B�?�@�B�B�B�B�B�B�����������������������y�s�j�g�m�s���	�"�$�,�.�+�(�"��	���׾̾������Ⱦ׿	�����������������������������������������/�;�?�;�9�/�"������"�$�/�/�/�/�/�/�����!�'�.�:�G�Q�G�:�.�!�������������"�/�;�E�J�L�J�:�"��	�������������������)�5�5�)����� ����������(�*�4�5�4�0�+�(������������(�4�A�Z�f�s������������f�M�3�#���#�(���'�-�3�7�=�3�)�'�����������y�������������y�p�w�y�y�y�y�y�y�y�y�y�yìùü��������ùóìàÓÇÅÃÇÑÓàì�5�A�N�Z�g�k�s�v�s�c�Z�A�5�(����(�,�5����������������������������������������²¿����������¿²¦�|¦²�B�N�[�g�k�j�h�g�\�[�N�B�9�5�+�2�5�@�B�B������������������	������������������"�'�/�1�0�'� ��	�����������������~�������������ź����������~�r�j�f�i�r�~ŔŠŧŭűųŭūŠŔœŇł�{�x�w�{ŇœŔ�����)�4�C�N�V�M�D�4�'�����������Ƨ������������� ������ƳƚƏ�w�c�\�hƚƧǭǡǕǔǓǈ�{�p�x�{ǈǎǔǡǭǲǳǭǭǭ�y������������������y�v�y�z�y�v�y�y�y�y�`�l�q�r�m�l�d�`�S�G�E�F�G�N�S�]�`�`�`�`�b�n�{�{�~�{�n�b�U�K�U�Z�b�b�b�b�b�b�b�b�����ûлܻ���ܻڻлû���������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��*�6�C�O�[�\�O�L�C�6�*�'� �&�*�*�*�*�*�*E7ECEPE\E_EcE\EPECE8E7E-E7E7E7E7E7E7E7E7EuE�E�E�E�E�E�E�E�E�E�E�E�E�EuEtEnEmEuEu % i W . + ^ � A 0  T " 5 ] R 5 ) = " u 4 > E . ]   \ �  7 [ i x / 6 O : k e @ > < [ , h k > 9 ' N @ f [ D )  5 N    �  �  �  �  �  O  �  �  u  �  �  �  �  (  �  [    �  �  �    "  e  {  Q  '  S  `  �  l  �  M  �  '  �  �  �  �  �  9  $  �    M  &  S    7  5  �  �    �  �  w    U  �  �  Q  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  N  ?  5  (      �  �  �  �  t  K     �  �  �  c  0  �  �  �  �  �  �  �  �  �  �  v  i  Y  I  8  )      	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  I  -  	  �  �  -   �   j  9  6  2  .  )  $                 �      -  E  ^  w  �  �  �  �  �  z  n  a  Q  ?  ,      �  �  �  �  �  }  T  �  �  �  �  �  �  �  n  K  (    �  �  �  ]  2     �   �   |  �  �  �  �  �  �  �  �  �  �    #  	  �  �  {  )  �  H  �  �  �  .  L  _  t  �  �  �  �  �  �  �  x  h  W  C  /      �  �  �  �  �  �  �  �  �  �  �  }  z  k  Y  <       �  �    F  r  �  �  �  �  v  ^  :    �  �  b    �  �  '  J  �  -  "      �  �  �  �  �  �  �  t  `  L  6       �   �   �  �  �  !  D  O  I  ?  4  '      �  �  �  �  �  o  L  !  �  �  �  �  �  �  �    j  S  :    �  �  �  L  �  �  -  �  {  �  �  �  �  �  �  �  �  �  �  �  �  �  �              �  �  	z  	�  
%  
M  
`  
c  
M  
&  	�  	�  	h  	  �  �  �  �  �  �    =  p  �  �  �  �      �  �  �  �  �  a    �  `  �  �                �  �  �  �  �  m  N  )    �  �  p  %  �  �  �  �  �  �  �  �  �  �  �  �  S    �  x  �  F  �   �  ^  u  �  �  �  �  �  �  u  e  K  %  �  �  �  v  3  �  �  \  j  e  _  Z  T  R  \  e  n  w  �  �  �  �  �    !  >  \  y  $  5  B  J  H  <  )    �  �  �  �  {  T  (  �  �  O  �  �  
�  
�  
�  
�  
�  
�  
y  
@  	�  	�  	3  �  N  �  y    �  �  :  _  �  �    '  D  Y  e  h  `  L  -    �  |    �  )  �  �  9  �  �  �  �  �  u  F    �  d    
�  
%  	�  	
  P  Z    Q  �    �  �  �  �  �  v  l  g  `  X  K  9  !    �  �  V  �  _  -  c  �  G  �  �  F  �  �      �  }  �  B  c  A  
�  P    V  j  v  p  d  M  /    �  �  o  6  �  �  y  8  �  �     �  �  �  �    1  B  R  c  v  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  ~  F    �  �  f  -   �   �   ~  s  �  �  �  �  �  �  �  x  \  :    �  �  {  8  �  �    �  y  �  �  �  �  �  �  �  �  �  �  o  E    �  L  �  f  �  v  �  �  �  �  �  �  �  �  }  t  i  ]  P  D  7  ,  !        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  5  p  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  `  J  6  "    �  �  �  �  �  �  �  �  �  �  �  �  �  X  	  �  �  4  @  3  1  %        �  �  �  �  �  �  �  �  �  u  a  L  1    �  �  �  �  �  �  �  �  �  �  �  �  x  p  h  _  W  O  F  >  6  �  �  �  �  �  �  �  d  ?    �  �  �  m  9  �  �  &  �  o  T  T  V  T  K  D  t  �  �  �  �  �  �  N  �  �  �  E    �  �  �  �  �  ~  u  l  b  W  L  A  6  +  $  #  !           �  7  Q    �  �  �  �  �  ]  "  �  ~    x  �  �  �  �  �  w  �  �  �  �  �  }  h  O  1    �  �  Z    �  �  _    U  �  �            �  �  �  �  d  %  �  �  +  �  k  �   �  g  ]  T  `  I  0    �  �  �  �  �  �  y  g  M    �  s    �  �  �  �  �  ~  e  H  +  
  �  �  �  �  z  g  V  H  ;  .  �  �  �  �  �  �  m  B    �  �  f  3  	  �  �  U  L  �  ?  P  E  /    �  �  �  �  m  J  #  �  �  r     �  r  �  %  �  (  R  u  �  �  �    i  F    �  �  �  [    �    x  �  �  D  6  %    �  �  �  �    U  2    �  �  �  l  :  	  �  v  e  N  1    �  �  �  �  N    �  t    �  T  �  �  &  (  �  i  V  H  9  !    �  �  o  6  �  �  �  >  �  �  N  �  w  �  �  �  �  r  b  N  =  &    �  �  �  k  2  �  �  $  �  \    �        �  �  �  �  �  �  b  D  &    �  �  �  �  z  r  r  j  a  S  D  ,    �  �  �  �  y  T  0    �  �  �  �  �  �  s  `  N  ;  &    �  �  �  �  �  l  Q  6      �  �  �  	�  {    �  �  '  G  W  Z  J    �  D  �  
�  
+  	  �  �    +  B  �  �    �  �    5     �  @  �  �  �  a  L     0  	�  f  X  H  2      �  �  �  �  U  #  �  �  Q  �  �  (  �  K  �  �  d  5  
  �  �  �  V  %  �  �  �  t  T  4  �  N  �  C  	�  	�  	�  	�  	�  	�  	�  	y  	A  �  �  4  �  8  �  .  �    �  /