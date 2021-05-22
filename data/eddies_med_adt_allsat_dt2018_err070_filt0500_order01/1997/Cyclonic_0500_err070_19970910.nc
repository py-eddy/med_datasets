CDF       
      obs    J   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�p��
=q     (  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N͠   max       P��(     (  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��1   max       <u     (  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F\(��     �  !$   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vQ�����     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @O�           �  8D   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�          (  8�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       �o     (  :    latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�K$   max       B4��     (  ;(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�z   max       B4�M     (  <P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�Q(   max       C�c#     (  =x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >w��   max       C�X|     (  >�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          s     (  ?�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     (  @�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7     (  B   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N͠   max       Pz5�     (  C@   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��-V   max       ?�i�B���     (  Dh   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       <D��     (  E�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F�Q��     �  F�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @vQ�����     �  RH   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P@           �  ]�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�=@         (  ^l   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D   max         D     (  _�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?y�_o�    max       ?�T`�d��        `�   #         &                  /   s            	         	                  	         #                     (         #         	                     I         '                           P         <                   	      6      &      Oā�O'}NMZ>P�O�jO �4Ni�O��OjjO�]oP��(O7�N�N3w�O*p;NZ*�N^� N���O&�YN��8O"�WN6��PSN�֭N��QN.&Pz5�O[�zOq�N͠O��VNG4uN��lP"L�O �OYAO�N�N��\O���N�XlN���N�1O��NT1�O4n�Oe�PP,�MN��!N�r�Oڈ�O/zN$��O7�ND�N��3N�֏Nt�O�<OكLO��SO9�<P!��Ocn-N�9�N��IO$+�O ��N�;8Ow��O�ӿO��O΄OoH�N�I�<u<o;�o:�o%   ��o�o�o�o�49X�49X�D���D���e`B��o��o��t���t����㼣�
��j��j��j��j�ě�����������/��`B��`B��h��h�������o�o�+�C��C��C��t���w�#�
�#�
�#�
�#�
�'49X�@��@��@��L�ͽL�ͽP�`�P�`�aG��e`B�ixսixսixսixսm�h�y�#�y�#��7L��\)��\)��\)���������
���
��1����������������������������������������tz��������zytttttttt)5BNfmnke[NB)��������������������������

�������
#$+#"









!#"�����������������������)6BV[[cVMB6)�#0Ifv{~sb<230#
���~����������������z~��������������������
#(/2/#!
��������������������������������������NO[hntxtkh[SONNNNNN�������������������������������������������������������������������������Y[\hqtvtlh[ZYYYYYYYY������� ����������PUYblnouxwnbURPPPPPPGO[`hjkhfhih[YROIGGG��������������������
#0<b������{b<0

�����

������kw�������������zrmik��������������������#/<UekksnaUH</#  #JOV[dhjhg[OCJJJJJJJJ4;AHNTYXTSKH?;3/4444�����/0
��������OOX[ht{utlllh[WSQOMO������������������)6O[bhmnmlsv[OE6'��������������������$)9BN[gtz�~tg[OB5'$)03+)	�,0<FIRUI<0-*,,,,,,,,��������������������agikmz��������zmc_ca�	�����������@BIN[^dknt{tg[NB;9=@�����������������������3?6)������������

����������������������������������������������������������������������������������������{ytg^[XUSUW[^gtuz��>BDO[\hhh[ZOIB>>>>>>������������������������������������������������������������������������������������)5IJB)������T[ft������������tdQT�����������������������
<NPY\ZR/#
������������������������rt}����������tllrrrr�������������������������

���������SUY^anzz~zrnaUURQS����������������������#)+*$������������������������������������ ���������")5BN[]^Z\^aSB5,��������������������./4:<CHMRURHE<2/--..�<�#�������������������#�/�A�J�Q�T�L�<�U�P�H�@�<�6�3�<�H�U�a�n�q�w�~�}�z�n�a�U�M�D�A�?�<�A�M�S�Z�\�_�Z�M�M�M�M�M�M�M�M�A�<�5�����5�A�N�g�s�~�������s�g�N�A�A�:�5�.�(�(�*�5�A�N�Z�s���������s�g�Z�A�s�h�f�f�f�q�s�����������������������s�н̽Ͻнݽ������ݽннннннннк��ֺϺϺֺ�������!�+�(�!������y�w�m�`�]�V�Q�T�`�m�y�����������������y�x�S�A�5�1�3�F�S�_���������������������x�����q�o�y�����ݾ�A�Y�q�f�A���ȽĽ������������������%�*�(�$���������ݽнʽĽýĽнҽݽ�������������������������
���
���������������׾ʾ��������������������ʾ׾������׾s�n�h�s���������������s�s�s�s�s�s�s�s�m�j�b�c�`�]�`�m�r�y�}�{�y�u�m�m�m�m�m�m�������������������������ƾƾ����������������(�4�A�E�K�J�A�:�4�(�������Ŀ��������������ĿʿѿҿԿӿѿƿĿĿĿ��������������������
�����
����������'�����#�'�,�3�5�8�3�'�'�'�'�'�'�'�'�3�2�9�<�3�,�9�@�Y�r�����������~�r�e�@�3�������������������ʼϼҼͼʼ�������������ܾ�������	������	���������
���'�4�@�F�@�4�*�'������������o�c�]�E�%��(�N�������������������������z�m�f�a�l�z��������������������������������������������$�+�+�*�'�$����Z�Y�Y�Z�g�r�s�y�s�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z���������������	�������������������ÓÍÇÃÇËÓÞàæäàÓÓÓÓÓÓÓÓ�A�9�5�2�5�9�A�N�W�Z�g�i�g�f�Z�N�A�A�A�A�޺ɺϺκֺ��-�:�F�_�f�c�p�_�S�:���޹ùù��������ùϹйܹ����������ܹعϹ�àÕÏÈÆÇÉÓàìòù����������ùìà�s�g�m�u�{�����������������������������s�T�Q�G�;�;�6�7�;�G�T�^�`�h�d�`�U�T�T�T�T���ʾžžǾҾܾ����	�����������H�H�@�;�0�2�;�H�T�V�\�a�g�a�]�T�H�H�H�H�������	����&�������������������)�6�B�C�B�8�6�)�#��������ùìèÙìù������������������Ҿ׾̾Ծ׾����������׾׾׾׾׾׾׾����	���	��"�/�:�;�=�G�L�L�G�;�/�"��/�*�#�%�)�/�;�H�T�a�l�m�x�x�m�a�T�H�;�/���r�[�>�6�4�@�Y�f�h�p������ּ��ʼ���ùùìêëìíù����������������ùùùù���	����*�.�1�+�*����������������������z�a�H�D�C�G�T�a�m�z���������h�\�`�h�uƁƎƚƧƲƳƸƳƧƚƎƁ�u�h�h�û��ûлԻܻ����ܻлûûûûûûû�ƎƚƧƯƱƧƢƚƎƁ�u�h�^�\�S�[�h�uƁƎ�����������������������������������������*�"�������*�-�6�C�H�C�@�6�*�*�*�*��������������������������������������żŹŮŹ�������������������������������U�I�<�#��
���������#�<�I�b�j�o�n�c�U�h�^�Y�`�iāěĬĿ������������ĿĳĚā�hĿĻĳĭĪĨĢĦĿ��������������������ĿĳĦĚčā�t�h�nāčĚīĳĻĺĿ����Ŀĳ��ݿѿʿƿ̿ѿ�����(�>�E�E�>�(��������� ��	����(�5�<�A�?�:�5�*�(��V�R�I�D�=�9�=�I�V�b�c�o�t�w�o�b�V�V�V�V�0�/�$����	������$�(�*�0�6�3�0�0D�D�D�D�D�E EEEE*E,E.E*E(EEEEED��������ݾھ�����	��"�&�(�"�!��	���.�!�!� �!�.�:�B�G�S�`�c�i�a�`�S�G�:�.�.�����������������Ľнܽ���ݽнĽ������������������������������ûǻɻɻȻû����f�]�_�v���������Լ�����ڼ���������f�^�U�P�J�G�C�E�H�U�n�zÓàùþ��ù�v�n�^²ª¦²¿��������������������²D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� O , V  " N = n ' % Y < q ~ W i n 8  d ? ; R 3 H | Y 0 ; 4 I C % C P 5 W 0 0 X @ K g j ) $ t ( 4 : g ` N T ` r N N e 8 O  I a k V 4 N 3 D z D K ;    i  x  H  �  m  u  ]  I  �  �  �  }  �  �  o  �  �  `    ~  A  �  �  �  �  z  �  �    �  [  �    '  �  +  �  t  �  �  �  B  r  �  �  �  �  �  �  �  T  �  �  �  �  �  �  �  J  �  �  �  �  �  �  ]    �  1        ���
�o�o��P����e`B�ě����㼛��m�h�   ���ͼu��1���ͼ��
���
��������/�t���`B�]/����h���y�#�0 ŽH�9�o�aG��t���P��\)�H�9�m�h��7L�49X�]/�0 Ž�w�'�o�<j�y�#�u��`B�8Q�P�`��{�y�#�P�`��7L�ixսY��q���u��O߾C����w�����h��Q콋C���7L�Ƨ�-������-�J���ͽ�����`��B"eB!RB�eB��BpB#JwB?�B7=B*�iB��B&XgB��B"*�B]1B<gB�B�]B4��B�B-�B^�B�)B!��B'�{B%uB �B'LBQ#B ��B �UB�B,A�K$B"�B�pB5�B|�B!�\Bt-B��B&I�B�>A�Z:BP\B��B�jB3oB��B7�B��B��B)Y*B	V�B��Bf�B�BåBBU$B	�>B}CB�B�NB
W�B;%B�B�B�\B�B��B,$�B�Bv+B�sB?�B!?�B�[B�BH�B#>�B?�B?�B*��B��B'>�B B"D�B@\B?�BE0B�B4�MB:gB-��B8�B��B"BrB'ȬB6
B ?=B&�9B�JB ��B ��BAB=LA�zB"�=B|�B�_B�qB!C�B	-SB?�B%�WB8tA���B#�BC�B��BEBB@zB|jB�gB?�B)?�B	@�BҾB�B��B;�B]�B?�B	�/B�%B4XBAB
�BO�BB�BR�BD�B?�B� B,=�BA6B��B��A���A�nA<�SA���A��kAE��A+'@TO5Al_�@��A,=YA1m�A*Q�A� �AR?AE�;AjB�AL��A6��Axr�A��X?���?�\+@��OAX��@��A��SA��B� A���A�y�A��;A�{e@e��>�Q(A�{A�H$Af%AW*A���@�>�A�_�A�EgAU�A�/4A��:@���A��uA�^<A��~B�k@���BR�A�a<A���A�6�A��dA�H�A��A��A���A���A�1OB�QB	�[C�c#AY�#A{A$α@���@�.0A�!A��.C��A���A��A= A�C�A�zpAEwgA+�@L!Al��@�2�A-"[A22SA*BLA��ASb�ADtAj�ALЀA73�Ay�A撍?���?��f@��yAX�]@�
QA���A��bBΠA��hA��Aʊ�A�|M@cS.>w��A̎�A�~
Af�AX��A�~�@���A� �A�|�ATk4A���A�WL@��ÀA�u1A���B�d@���B��A���A�ښA��}A�z�A�A�&A�WA�m�A��ZA�{�BDB	A�C�X|A[)�A�A&��@�a@�'�A�Q�A�pCC�]   $         '                  0   s            
         	                   	         $                     )         $         
                     I         '                  	         P         <   !               
      7      '          !         %                  #   ?                                    )            7            !         1         '                              7         %                        !   '         '                           '   #                  !                     /                                    #            7                     !                                       1                                             %                           !   #      O��O�gNMZ>O�]O��NjUGNi�O��O oO ��P� O7�N�N	#�O�>NZ*�N^� N�S�O&�YN��8O"�WN6��O�+ON�֭N��QN.&Pz5�O8�	Oq�N͠O�}�N$3N��lO��NF� O82ZOvq�N��\O���N�XlN���N�1O��NT1�O)e�Oe�PPL�N��!N�r�OxfO/zN$��O7�ND�N��3N�֏Nt�O�|�O�u�O{�N���Pe"Ocn-N�9�N��IO$+�O ��N�;8Ow��O�!sOw�O΄O[��N�I�  �  O  �  �  �  �  B     �  �  	�  "  �    r    "  .  H  �    �  -  A  �  (      �  u  
  M  Y     �  p  c  �  �  R  l  �  �  �  *  S  
d  �  �  W  w  �  *  u     �  h  �  �    �  q  �  �  �  	�    B  |  	�  �  K  �  	
<D��;�`B;�o��o��o�ě��o�o�t��+�P�`�D���D���u��C���o��t����㼛�㼣�
��j��j��`B��j�ě�����������h��`B��`B�o������w��w�\)�,1�+�C��C��C��t���w�#�
�'#�
�<j�'49X�m�h�@��@��L�ͽL�ͽP�`�P�`�aG��ixս�hs����q����o�m�h�y�#�y�#��7L��\)��\)��\)�������㽣�
���T��1����������������������������������������tz��������zytttttttt%)5BN[bijf`[B5)!%��������������������������������������
#$+#"









!#"�����������������������!)46BNOQQOGB6)'#0IWdnqnbUID<60#~����������������z~��������������������
#'/0/#"
��������������������������������������NO[hntxtkh[SONNNNNN�������������������������������������������������������������������������Y[\hqtvtlh[ZYYYYYYYY��������������������PUYblnouxwnbURPPPPPPGO[`hjkhfhih[YROIGGG��������������������
#0<b������{b<0

�����

������kw�������������zrmik��������������������"#)/<HUcgifaU</$""KOW[ahihd[OHKKKKKKKK4;AHNTYXTSKH?;3/4444�����
���������T[ghhtnh_[ZRTTTTTTTT������ �������������')6BO[chkja[OH=6/*''��������������������$)9BN[gtz�~tg[OB5'$)03+)	�,0<FIRUI<0-*,,,,,,,,��������������������agikmz��������zmc_ca�	�����������@BJN[]cjmsgc[NFB<:=@�����������������������.58 ������������

�����������������������������������������������������������������������������������������{ytg^[XUSUW[^gtuz��>BDO[\hhh[ZOIB>>>>>>�����������������������������������������������������������������������������������#*0)�������agst���������tmgd_aa�����������������������
1<ANVSI/#
������������������������rt}����������tllrrrr�������������������������

���������SUY^anzz~zrnaUURQS����������������������#)+*$����������������������������������������������")5BN[]^Z\^aSB5,��������������������./4:<CHMRURHE<2/--..�
���������������
�#�/�<�G�N�Q�H�<�#��
�U�Q�H�A�<�7�8�<�H�U�a�n�p�v�}�|�z�n�`�U�M�D�A�?�<�A�M�S�Z�\�_�Z�M�M�M�M�M�M�M�M�(�$���"�-�5�A�N�g�s�y���������s�Z�A�(�N�A�3�0�.�2�5�A�N�g�s�����������s�g�Z�N�s�r�l�s������������������s�s�s�s�s�s�н̽Ͻнݽ������ݽннннннннк��ֺϺϺֺ�������!�+�(�!������`�_�W�T�S�T�`�m�y���������������y�m�`�`�l�j�_�O�F�E�F�S�U�_�l�x�����������{�x�l�������������ݽ���7�D�G�A�)���ʽ��������������������%�*�(�$���������ݽнʽĽýĽнҽݽ������������������������	�
��
�
�� �������������׾ʾ��������������ʾ׾�������ھ׾s�n�h�s���������������s�s�s�s�s�s�s�s�m�j�b�c�`�]�`�m�r�y�}�{�y�u�m�m�m�m�m�m���������������������¾ľ��������������������(�4�A�E�K�J�A�:�4�(�������Ŀ��������������ĿʿѿҿԿӿѿƿĿĿĿ��������������������
�����
����������'�����#�'�,�3�5�8�3�'�'�'�'�'�'�'�'�@�7�=�@�>�6�@�L�r�~���������~�r�j�_�L�@�������������������ʼϼҼͼʼ�������������ܾ�������	������	���������
���'�4�@�F�@�4�*�'������������o�c�]�E�%��(�N�������������������������z�m�m�d�m�p�z������������������������������������������$�+�+�*�'�$����Z�Y�Y�Z�g�r�s�y�s�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z������������������������	�������������ÓÏÇÄÇÎÓÚàäâàÓÓÓÓÓÓÓÓ�A�9�5�2�5�9�A�N�W�Z�g�i�g�f�Z�N�A�A�A�A��غٺߺ����!�-�:�K�S�Z�S�:�!������ù��������ùϹعܹ�ܹϹùùùùùùù�àØÓËÉÌÓàìíù������������ùìà���|�w�{�}�������������������������������T�Q�G�;�;�6�7�;�G�T�^�`�h�d�`�U�T�T�T�T���ʾžžǾҾܾ����	�����������H�H�@�;�0�2�;�H�T�V�\�a�g�a�]�T�H�H�H�H�������	����&�������������������)�6�B�C�B�8�6�)�#��������ùìèÙìù������������������Ҿ׾̾Ծ׾����������׾׾׾׾׾׾׾����	���	��"�/�;�F�H�K�K�H�F�;�/�"��/�*�#�%�)�/�;�H�T�a�l�m�x�x�m�a�T�H�;�/���r�]�@�9�7�@�M�Y�l�{�������ּ�ټ�����ùùìêëìíù����������������ùùùù���	����*�.�1�+�*������������z�o�m�a�V�\�a�m�z���������������������h�\�`�h�uƁƎƚƧƲƳƸƳƧƚƎƁ�u�h�h�û��ûлԻܻ����ܻлûûûûûûû�ƎƚƧƯƱƧƢƚƎƁ�u�h�^�\�S�[�h�uƁƎ�����������������������������������������*�"�������*�-�6�C�H�C�@�6�*�*�*�*��������������������������������������żŹŮŹ�������������������������������U�I�<�+�#��
����
��#�<�I�b�h�m�k�b�U�h�b�e�nāčĚĦĳĿ������ľĳħĚā�t�hĿĻĳĳıĳĿ����������������������ĿĿĚďčā�t�l�p�tāčĖĚĦħĳĴĳĦĥĚ��ݿֿοʿѿ׿ݿ����(�9�@�@�9�(��������� ��	����(�5�<�A�?�:�5�*�(��V�R�I�D�=�9�=�I�V�b�c�o�t�w�o�b�V�V�V�V�0�/�$����	������$�(�*�0�6�3�0�0D�D�D�D�D�E EEEE*E,E.E*E(EEEEED��������ݾھ�����	��"�&�(�"�!��	���.�!�!� �!�.�:�B�G�S�`�c�i�a�`�S�G�:�.�.�����������������Ľнܽ���ݽнĽ������������������������������ûƻȻǻƻû����e�b�f�l�x�������ʼּݼռʼ���������r�e�^�U�P�J�G�C�E�H�U�n�zÓàùþ��ù�v�n�^²¬¦²¿������������������¿²D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� N * V  ! J = n )  f < q � ! i n /  d ? ; U 3 H | Y 1 ; 4 A 9 % > 7 9 > 0 0 X @ K g j # $ t ( 4 3 g ` N T ` r N K M  =  I a k V 4 N 3 A l D E ;  �  P  x  �  #  �  u  ]    \  $  �  }  �  '  o  �  �  `    ~  A    �  �  �  z  �  �    u  >  �  �  ]  �    �  t  �  �  �  B  r  l  �  `  �  �  �  �  T  �  �  �  �  �  D  p      D  �  �  �  �  ]    �     h    �    D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  �  �  �  �  �  �  �  �  �  �  n  Y  7  �  �  0  �      R  9  L  H  B  ;  ,      �  �  �  j  5  �  �  N  �  t   �   �  �  �  �  �  �  �  �  �  �  �  y  c  M  4    �  �  �  �  V  W  �  �  �  �  �  �  u  a  O  I  5    �  �  g    �  �   r  X  n    �  �  �  {  g  K  *    �  �  y  C    �  �  �   �  Z  c  h  k  l  m  o  �  �  �  �  }  b  @     �  �  �  z  ,  B  9  0  '          �  �  �  �  �  �  �  �  �  |  d  L     �  �  �  �  �  �  �  �  {  t  S  .    �  �  �  �  �  �  �  �  �  �  �  �  x  m  b  V  J  >  3    �  �  �  _     �  �  �  �    /  R  f  u  �  �  v  U  -    �  �  -  �  c  �  �  A  �  	#  	y  	�  	�  	�  	�  	�  	c  	  �    d    �  �  �  �  "      �  �  �  �  �  �  �  �  �  k  M  *    �  �  �  b  �  �  �  �  �  �  �  �  �  �  �  �  z  n  `  Q  C  5  &            	    �  �  �  �  �  �  ~  g  O  5       �  �  U  c  p  d  T  ?  )    �  �  �  �  �  �  �    ~  z  W  5      	    �  �  �  �  �  �  �  �  �  y  a  H  .     �   �  "                
                             *  ,  *  %        �  �  �  �  �    [  4     �   �  H  G  D  @  <  4  ,  !      �  �  �  P    �  �    �  _  �  �  ~  u  i  ^  R  E  8  (      �  �  �  �  i  N  8  !    �  �  �  �  �  v  T  9  !  	  �  �  �  �  �  �  }  Y  2  �  �  �  �  �  �  �  �  �  |  x  s  ^  ?        �  �  q  @  �    '  -  &      �  �  �  �  �  u  ?  �  �  |  B  "  ?  A  >  ;  4  *      	  �  �  �  �  �  �  z  X  4    �  �  �  �  �  �  �  �  �  |  t  m  f  _  W  O  G  ?  1  !      (             �  �  �  �  �  �  �  )  �  u  @     �   �      �  �  �  ]  &  �  �  f  (  �  �  v  1  �  �  �  X   �  �  �       �  �  �  �  �  �  x  T  .    �  �  �  O       �  {  q  b  K  0    �  �  �  �  P    �  �  E  �  �  w  �  u  g  X  J  ;  ,      �  �  �  �  �  �  j  X  G  6  %    �  �  �  �  �  �  �  �  u  d  I  #  �  �  �  u  B    �  �  H  J  K  L  I  E  =  1  %    �  �  �  �  _  .  �  �  �  ^  Y  P  F  =  2  (      �  �  �  �  �  �  �  �  u  f  W  I  �  �  �  �  �  �  �  �  �  �  �  |  a  D    �  |  �  E    �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  W  <    �  �  f  n  p  o  l  g  ]  H  )  �  �  �  h  8    �  �    \  �  �    4  T  b  c  _  Z  J  (  �  �  {  3  �  <  �  0  7   ?  �  �  �  �  �  �  w  e  Y  N  D  ;  1  &            �  �  �  �  �  �  �  �  }  c  F  $  �  �  �  �  k  +  �  �    R  D  7  &       �  �  �  �  s  R  /  �  �  p  E     �   �  l  k  k  k  i  c  \  V  M  A  5  )    	  �  �     #  E  h  �  �  �  �  �  �  x  m  _  M  <  *        �  �  �  �  �  �  �  �  �  �  i  A    �  �  _    �  k    �  )  �  8   �  �  y  e  R  8      �  �  �  �  r  V  9    �  �  �  �  �  �  )      �  �  �  �  k  <    �  �  b    �  n    %    S  N  I  C  ;  2  )        �  �  �  �    T  %  �  r    	�  
]  
b  
P  
/  
  	�  	�  	[  	  �  t  >  �  /  p  �  �  �  �  �  �  }  i  U  ?  $  
  �  �  �  �  �  x  b  H  -     �   �  �  �  �  �  �    q  b  T  C  2       �  �  �  �  ~  a  C  ;  <  B  L  T  U  V  T  J  7    �  �  �  _    �    0  7  w  `  G  .    �  �  �  �  �  t  c  u  g  >  �  �  [  �  �  �  �  �  �  �  �  �  �  �  �  x  n  c  Y  N  :  !  	   �   �  *    �  �  �  �  �  |  \  8    �  �  s  .  �  _  �  �  �  u  `  L  7     	  �  �  �  �  �  d  C  !  �  �  �  ^     �                           	          �  �  �  �  �  �  s  a  O  9  $    �  �  �  �  m  N  *    �  �  �  h  ]  R  G  <  0  %      �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  n  R  2    �  �  �  [  $  �  �  {  :  �  &  X  �  �  �  i  A    �  ^  �  a  
�  
#  	_  �  �  �  ^  �  �  �  �  �  �  �      �  �  �  �  }  H    �  �  d  �  !  �  �  �  �  h  F  %  �  �  �  t  5  �  �  H  �  b   �   �  U  j  q  o  `  I  +    �  �  �  ~  M    �  S  �  P  �  E  �  �  �  �  �  �  t  V  3    �  �  S  �  �  $  �  -  �  S  �  �  �  �  �  �  |  l  [  I  6  $    �  �  �  �  �  u  S  �  �  �  �  �  �  �  �  �  p  V  8       �  �  �  �  �  �  	�  	�  	\  	$  �  �  V  �  �  ,  �  I  �  �  &    �  B  �  �    �  �  �  �  �  ~  X  3    �  �  �  S  #  �  �  �  `  j  B  1     	  �  �  �  �  �  v  d  U  J  A  <  3       �  �  |  u  m  e  Z  M  <  *      �  �  �  �  i  =  �  �   �   g  	�  	�  	�  	�  	�  	�  	�  	[  	  �  q    �  Z  �  k  �  �  �  �  s  Y  �  p  O  +    �  �  �  ]  3  �  �  �  6  �  �  �    K  !  �  �  �  �  �  q  R  1  �  �  r    �  6  �  *  �    �  �  �  �  g  C    �  �  �  ]    �  x     �  u    �   �  	
  	  �  �  �  �  W    �  v    �  T  �  n  �  f  �  B  �