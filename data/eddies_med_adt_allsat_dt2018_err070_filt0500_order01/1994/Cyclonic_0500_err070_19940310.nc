CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��O�;dZ       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       P���       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��hs   max       <e`B       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?������   max       @Fj=p��
     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���P    max       @v�(�\     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P            �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��S�   max       <49X       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B1;'       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��V   max       B0��       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�80   max       C��       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��3   max       C��       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ]       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�9�       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�C,�zxm   max       ?ҹ�Y��}       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��hs   max       <e`B       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�ffffg   max       @Fj=p��
     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�fffff     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P            �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�o            Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?   max         ?       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?ҹ�Y��}     �  ]         
      	                     X      ?                              '   !      	      \            /                     
         ,                           	         '                           (   	               N
�iN:r�Ns"�P&Nv�OB�KN;�:Nw��N�N˫�O>s�P��OIO��N�g�O�DO���N�O�O���O��NW�O'xwO���O�4�PZ��N���NO��O��BP���P��O�ȕO�3�O���O׆O��Nς�N8��OR1<O�Y�N�t7P��O}��P�?O�A�OQM�O ACNx�O�m�O#TN2��N�lZN�5�N��O4�O��N�6�OP�Ofv�O ��O��8N7ŨN���O@�O���N�%�Non�OW�#OT�BNp��Nu�g<e`B;�`B;�`B;ě�:�o$�  ��o��o��o���
�ě���`B�o�o�t��#�
�49X�D���D����o��C���C���C���t���t����㼛�㼛�㼛�㼛�㼬1��1��9X���ͼ��ͼ�/��/��/��/��`B��h��h���o�o�C��t��t��8Q�8Q�8Q�<j�@��@��@��@��@��@��D���H�9�H�9�Y��aG��e`B�u�u��+��\)��\)��hs��������������������tz�������|ztttttttt,0<IUU]UOI<<20,,,,,,�������������������������������~��������),59>A?5)
���
����������)+0-)����������������������������������������������
	���������������#28#�����;=HUamntwnknona^UH@;uz�������������xsqu�������


��������������������������
/Uc`UQHD</#
���(����������)5ABIN[glt~~wtg[B4)����������������������������������������Y[gt����������tgc[YYZgt������������tg[XZSXamz����������zmbTS������5>?64)������#$/6<HKHC=<5/#"��������������������s����������������{ls#<U�������{I0
����������������������������������������������������<0#
��#/9<?DHD<��������������������")069BFJKLEB6)'$#!"��������������������U[^hktzutmhe[UUUUUU�����������������~~�%*6CO_hw��uh\C6*% %��������������������GNgt��������{tg[NGEG&,69BEKSURKB;)/7@HLZp����znaUH<5//���������������������������������������NO[hmnkotwvthe[WOKJN<BNN[bc[URNIBA<<<<<<y������������tznolny>BN[gqtwtrgc[NDBA@A>���
����������))67BEIGB6+)#))))))KN[aghhg_[NGGIKKKKKK��������������������tz���������������zt�������������������������

	��������
#%+./'#
����#0<IUYZZUQI<0#��
"
	 ����ght������������}tmjg��������������������!#0<>INMI<0-&#!!!!!!DHUaiyz��zsnaYUJDB?D��"#  ��������������������������(/<?HHH<:3/+&$((((((������������egqt������������tgce��������������������4<BHRTHC<93044444444���������������������������������������������������������������޼������������ʼҼּټּּʼ��������������������M�4�,�1�N�Z�f����������߾˾Ǿ�����������	����������������������y�m�c�`�Y�^�`�h�m�y�������������������y�[�T�N�B�?�B�N�[�g�r�g�b�[�[�[�[�[�[�[�[�=�4�0�)�.�0�=�I�Q�R�I�?�=�=�=�=�=�=�=�=�����(�1�4�A�A�A�4�(�(�'�������#������#�/�6�<�H�O�U�U�U�H�<�/�#�#�"������"�"�/�;�H�L�R�Q�K�H�<�;�/�"�ʼ����a�^�u������ּ��!�:�B�A�:�-��ʾA�4�(�(�+�4�7�A�M�R�Z�d�f�i�g�f�e�Y�M�A��ݼټ�����!�.�6�=�9�/� ����������������s�p�s���������������������������)�(�#��#�)�6�O�[�h�u�v�y�t�h�[�O�B�6�)��� �%�,�)�6�B�O�\�h�m�k�Y�V�O�B�<�)�����������������������������������������������s�f�`�]�c�f�s�������������������������������������������$�,�3�0�$�������	��������	���"�$�"� ���������������������	�
��������	�������������������������ѿ߿��ҿпſ����s�Z�I�F�E�H�H�N�Z�g�s�z���������������s�N�8�9�3�(������(�N�s�������k�g�l�g�N�H�=�<�2�/�)�(�/�5�<�H�J�U�W�V�U�H�H�H�HFFFFF#F$F*F1F6F1F,F$FFFFFFFF�M�I�H�M�O�L�M�Y�f����������������r�Y�M���g�P�=�4��5�g�������������������������������p�f�e�o�t�������ĿԿڿ����ݿĿ��a�N�9�(���"�5�N�^�g�s�������������s�aÓÄ�r�g�c�i�w�zÇÓÜàâìô÷ñìàÓ�t�t�g�N�B�5�%���)�-�5�B�[�g�t�����������������������������������������f�\�f�o�s�w�����������������������s�fƎƌƁ�u�h�f�h�q�uƁƁƎƚƦơƧƬƧƚƎ�Ľ��������������ĽŽĽ½Ľ̽ĽĽĽĽĽĻw�l�g�`�i�f�e�l�x���������������������w�	����������"�.�7�:�;�7�5�2�.�"��	ŠŕŔňŋŔŘŠŧŭŹź����ŹŭŠŠŠŠ�������������������	��5�B�N�A�0������޿��������y�c�`�K�T�`�m�������������¿���F$FE�E�E�E�F1FcF|F�F�F�F�F�F�F}FhFVF=F$�������������)�J�T�Z�O�C�6�*��!���3�*�/�?�G�L�S�U�Y�r�~���z����r�Y�L�9�3�ù¹����ùϹܹ�������������ܹϹ��������������������������������������ŔŐŕŕšŹ����������������������ŹŭŔ����������������&�)�0�4�)�!������f�`�Y�V�Y�f�l�r�u�y�v�r�f�f�f�f�f�f�f�f�f�f�e�f�m�r�������������r�f�f�f�f�f�f�<�8�7�<�G�H�U�V�a�e�l�a�U�H�<�<�<�<�<�<�<�8�8�<�H�U�[�\�U�H�<�<�<�<�<�<�<�<�<�<�����������������������������������������ɺº������ɺԺ����#�5�?�@�:�-�����ɽ���ݽнɽʽнݽ������� ��������лϻû��������������ûлܻ߻���ܻлн����������������������Ľн۽нȽýý���E*EE E*E2E7ECEPE\EiEuEvEjEiE\EPECE;E7E*¦ ¦º¿����������������¿¦���������
��"�!��
���������������������!��� �!�(�-�:�@�F�>�:�-�'�!�!�!�!�!�!���������������Ŀѿ׿ݿ��ݿڿѿĿ������G�.�$���.�S�����нݽ�нȽ������y�`�G������!�$�0�1�1�0�&�$�������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ìèâââìòù��������������������ùìĳĮĦĥħįĳĻĿ������������������Ŀĳ����������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� � 3 Z \ .  W " v V . N 3 ^ e % _ 9 ( : i S E @ P a R H 1 L P N $ _ U S { @ S 0 / _ _ K q ] S , ) G 7 3 9 L Y s < 6 g & T Y @ s \ E  8 3 :  W  Z  �  (  {  �  _  y  2    �  �  b  L  �  6  �  �  ;  (  �  �  �  O  �  �  �  s  �  �  �  R  g  �  Y    {  �  g  �  g  I  d  �  <  <  �  ~  ]  e  �  �  ;  k  �    ,  �  J  Z  S  �  �  [  �  �  �  �  �  �<49X;D���o��t��o��9X�49X�49X�o�#�
��1�\�o��\)�T���#�
�+���㼼j��󶼴9X���o�e`B�P�`���ͼ�`B�49X��S��D���L�ͽ<j��O߽\)�t������49X�@���P�u�#�
�����H�9�H�9�L�ͽ#�
�aG��m�h�T���ixս]/�T���T���� Ž]/�q����o���-����aG��u���w�ě���C���t��� Ž��t���{B+AB�_B&�{B!�WB ��BbBKUBaB�B9^B]=B-G�B��Br�B;�BXgBD�B��B�
B�%BkB	�B
��A��B��Bl�B�B�-B'OB+�yBt�B��B�?B:B�IB �qB�B B1;'B�B	�B��B+7BB!�B��BL�B
��B�uB�&B�B�,B
�IB��By(B#�gB$��B&�B��B
�(B�@B%��B5�B#4B�ZBt�B̗B
P*B-�B��B+=nB��B&�B!ȍB ��B?KB?�Bx�B6hBAoBD�B-@�B��B@+B@�B@�B�B@3B�~B��BA�B	ԦB
�YA��VB��B8�B;�B=~B'A6B+�BR�B�UB��B�uB��B �	B5�B �QB0��B�B	@�B?9B�,B�B �	B@B?�B
�jB��B��BˬB��B
�B��B�B#�B$��B%��B��B
��BƫB&DUBEPB��B��B@�BO�B
E�B]^B��@W�A��_@���AF%A�jiAm��A�<�B
�jA7�A�ޗA���Aw]A;C�A��A��A��A�l�AH�`AG"�B��A[� AZ��Av�pA�L�A���A�ďC���@�3{A�NAwV�A��A�Z�A��A�h�AG�B��A%"[@��EA^q�A�� A�D�ApW�C��A�j�?�ʨ>�80A��\A��Aԧ�@߹�@�ъA�AĄ~@�W@W�ZA.��@�e�A#�PC���A��A���@s�Aw� A�B	jC�+�A�e�A��A��C��@_�kAЈ�@��ACx�A�tAmA�S�B
��A5�{A�MvA���@���A;-�A�2A���A؀@A�/AH�\AC�B	�A] "AZ�<Av��A�|wA��AĀC��@�(%A�~AuA���A�x�A�)�A��@AJ�B��A$ .@�}A[7�A�g�A��DAt��=��3A�"{?娔>�>�A��0A��A�~�@�.?@�~�A��kA�y�@�@[;�A-W
@��A"�C���A�l�A�mw@t�Ax�A�B	FbC�(OA�z�A�y�A�>C���         
      
                     Y      ?                              '   "      
      ]            0                     
          -                           	         (                           )   	                           3                        A               #         '         !   #   3         !   7   -   #                              %      -   %                                                            -                              !                        7                                 !                  5   -                                 #      '                                                               '                  N
�iN:r�NPevO�РNJ�LO �GN;�:Nw��N�N˫�O��PY�/NL^�O5�-N�g�O�p�O�GN�O�O��_O���NW�O��O���O&X�O=�N���NO��O�UgP�9�P��OJ-�O�3�O���O׆O��Nς�N8��OR1<O�Y�N�t7O�+4Oc��O�p*OZY;OhN��XNx�OG�NOA�N2��N�lZN�5�M���O4�O@��N�6�N�+LOfv�N�CO��8N7ŨN���O4�)O�Q�N�%�Non�OW�#OT�BNp��Nu�g  �  `  %      c  �  �  �  �  �  �  �  n    4  `  �  x    M  �  "  e    c  �  �  �  �  \  �  �  �    �  �  L  �  ~  I  �  �  �    �    =  `  O  x  �  -  [  �  �  �  �  	S  	  �  e  v  2  7  D  �  L  �  �<e`B;�`B;ě���o$�  ���
��o��o��o���
�o��1���㼣�
�t��49X����D���T����t���C����㼋C��t���P���㼛�㼴9X��/�������1��`B���ͼ��ͼ�/��/��/��/��`B�+���\)��P�\)�\)�t��'<j�8Q�8Q�<j�D���@��ixս@��D���@��H�9�H�9�H�9�Y��e`B�q���u�u��+��\)��\)��hs��������������������tz�������|ztttttttt2<IRU[ULIA<422222222���������������������������������������� )158853)���
����������)+0-)�����������������������������������������������������������+(�������FHUZacbaUHHDFFFFFFFFwz~�������������}zyw�������


��������������������������#.<HLNJHB=:/-#�(����������-58CDKN[gjt|}{rg[B7-����������������������������������������Z[gkt���������tgf\[ZZgt������������tg[XZ_akmz�������zmda^]]_&)*,*)# #$/6<HKHC=<5/#"��������������������ot���������������~uo#<n{������{b<0
����������������������������������������������������	#/<AE@8/*#	��������������������")069BFJKLEB6)'$#!"��������������������U[^hktzutmhe[UUUUUU�����������������~~�%*6CO_hw��uh\C6*% %��������������������JNR[g��������xtg[PJJ'.6=BGOSSQIB:);HUcs����znaUH<601;����������������������������������������OO[hmmjntvthf[XPOLJO<BNN[bc[URNIBA<<<<<<qtu�������������|vtqABCGNV[gptvtqgb[NEBA���
����������))67BEIGB6+)#))))))KN[aghhg_[NGGIKKKKKK��������������������tz���������������zt�������������������������

	���������	
#),-%#
�����#0<IUYZZUQI<0#��
!
	  ����ght������������}tmjg��������������������!#0<>INMI<0-&#!!!!!!EHU_ahyz��zrnaZUKEAE�� " ������������������������(/<?HHH<:3/+&$((((((������������egqt������������tgce��������������������4<BHRTHC<93044444444���������������������������������������������������������������޼����������ʼмּؼּռʼ����������������Z�T�A�J�Z�f�s������ž�����������s�f�Z���������������������������������m�h�`�_�`�f�m�r�y�����������������y�m�m�[�T�N�B�?�B�N�[�g�r�g�b�[�[�[�[�[�[�[�[�=�4�0�)�.�0�=�I�Q�R�I�?�=�=�=�=�=�=�=�=�����(�1�4�A�A�A�4�(�(�'�������#������#�/�6�<�H�O�U�U�U�H�<�/�#�#�.�"������"�&�/�;�H�I�P�O�H�H�;�/�.��r�r�{���������!�1�:�;�6���ּ�����4�2�2�4�=�A�M�O�X�M�M�A�4�4�4�4�4�4�4�4����߼�����!�.�0�2�.�)������������������s�p�s���������������������������6�)�$�"� �$�)�6�B�O�[�h�t�u�x�t�h�O�B�6�)�(�'�)�.�3�6�B�O�Q�[�h�h�`�[�Q�O�E�6�)������������������������������������������������s�f�b�_�d�f�v����������������������������������������$�*�0�2�$�������	��������	���"�$�"� ����������������������	�	������	��������������������������ѿ߿��ҿпſ����g�c�Z�Y�[�g�i�s���������������������s�g����	�����(�5�7�A�N�N�L�A�@�5�(��H�=�<�2�/�)�(�/�5�<�H�J�U�W�V�U�H�H�H�HFFFFF#F$F*F1F6F1F,F$FFFFFFFF�Y�Q�K�J�O�R�O�S�Y�f��������������r�f�Y���s�T�B�/�-�4�g�������������������������������p�f�e�o�t�������ĿԿڿ����ݿĿ��s�g�\�Z�P�N�X�Z�g�s�����������������~�sÓÄ�r�g�c�i�w�zÇÓÜàâìô÷ñìàÓ�B�5�+�"�#�5�B�[�g�t�}�t�g�[�N�B�����������������������������������������f�\�f�o�s�w�����������������������s�fƎƌƁ�u�h�f�h�q�uƁƁƎƚƦơƧƬƧƚƎ�Ľ��������������ĽŽĽ½Ľ̽ĽĽĽĽĽĻw�l�g�`�i�f�e�l�x���������������������w�	����������"�.�7�:�;�7�5�2�.�"��	ŠŕŔňŋŔŘŠŧŭŹź����ŹŭŠŠŠŠ�����������������������#�5�=�9�)����꿫�������y�g�`�W�`�h�y������������������FE�E�FF=FVFcFoF|F�F�F�F�F�FzFfFVF=F$F������������
��� �6�D�J�C�6�*����@�4�@�B�L�T�Y�Y�f�r�~��w�y�}�s�e�Y�L�@�ùù����ùϹܹ�������������ܹϹ��������������������������������������ŹűŭŠśŝŠũŭŹ������������������Ź�������������������%�)�/�2�)���f�`�Y�V�Y�f�l�r�u�y�v�r�f�f�f�f�f�f�f�f�f�f�e�f�m�r�������������r�f�f�f�f�f�f�<�8�7�<�G�H�U�V�a�e�l�a�U�H�<�<�<�<�<�<�<�:�;�<�H�U�V�X�U�H�<�<�<�<�<�<�<�<�<�<�������������������������������������������ֺҺк�������+�-�2�-� ���������ݽнɽʽнݽ������� ��������ܻ׻лû����������ûлܻݻ޻�߻ܻܻܻܽ����������������������Ľн۽нȽýý���E*E E!E*E2E7ECEPE\EiEtEiEiE\EPECE:E7E*E*¦ ¦º¿����������������¿¦���������
��"�!��
���������������������!��� �!�(�-�:�@�F�>�:�-�'�!�!�!�!�!�!�����������������Ŀѿֿݿ��ݿٿѿĿ����`�G�(��$�.�S�l���������ĽнŽ������l�`������!�$�0�1�1�0�&�$�������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ìèâââìòù��������������������ùìĳĮĦĥħįĳĻĿ������������������Ŀĳ����������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� � 3 P e .  W " v V ( G 6 X e & D 9 ) 6 i B E + / a R F * L  N % _ U S { @ S 0 & ] V 2 y ^ S  # G 7 3 5 L G s ? 6 d & T Y C j \ E  8 3 :  W  Z  �  ?  [    _  y  2    R  �  g  �  �    a  �    �  �  M  �  a  A  �  �  5  8  �  �  R    �  Y    {  �  g  �  �    �  �  �    �  �  1  e  �  �    k  �      �  0  Z  S  �  �  b  �  �  �  �  �  �  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  �  �  �  �  �  �  �  �  �  �  �  u  j  ^  R  F  :  .  !    `  [  U  P  J  D  :  0  &      �  �  �  �  �  �  �  u  a      $  !        �  �  �  �  �  �  �  k  ?    �  �  �  Y  �  �  �  �                 �  �  �  e  (  �  �  3  �  �      #  &  (  #      �  �  �  �  L    �  �  e  &  7  I  V  _  b  b  ]  Q  B  0    �  �  �  y  +  �  J  �  ?  �  �  �  �  |  f  N  6      �  �  �  �  v  S  /  �  �  v  �  �  �  �  �  �  �  �    n  ]  K  8  %    �  �  �  �  �  �  �    5  W  p  o  n  m  l  h  b  \  V  P  W  b  n  z  �  �  �  �  �  �  �  f  E  '    �  �  �  �  �  �  �  �  {  u  �  �  �  �  �  �  �  �  �  q  J    �  �  �  M    �  �  �  b  �  �  �  �  �  �  �  ~  b  <  �  �  $  �  �  *  m  �    -  �  �  �  �  �  �  �  �  �  �  �  �  Z    �  �  X    �  
�  4  W  h  n  `  B    
�  
�  
�  
Q  
  	�  	#  �  �  �  W  d        $  )  ,  (  #             �   �   �   �   �   �   �  1  2  +  %  !              �  f    �  �  w  q  �  �  �  =  P  J  H  O  ]  _  Q  2     �  z  5  �  �  e  P  <  �  �  �  �  �  �  �  �  v  g  P  7    �  �  �  �  �  @   �   �  q  u  x  w  r  j  ^  P  =  )    �  �  �  �  �  a  7  U  �                �  �  �  �  �  q  G    �  �  �  Q    M  K  I  G  B  5  )        �  �  �  �  �  �  �  �  l  V  g  |  �  �  �  �  �  p  [  A  #    �  �  �  �  ]  %  �  �  "  "        �  �  �  �  �  �  n  S  5    �  �  �  [    \  �  �    (  ;  J  W  c  e  \  G    �  g  �  �    n  �  5  I  U  J  �  �  r  o  �  �  �       �  �  }  -  �  L  �  c  �  �        "        �  �  �  �  }  \  B  F  K  P  �  �  t  ^  H  0    �  �  �  �  �  [  .  �  �  Q  �  �  F  �  �  �  �  �  �  �  �  u  A  
  �  �  �  �  �  B    �  ^  {  �  �  �  �  �  b  +  �  �  4  �  V  �  K  �  &  {  d   �  �  �  �  �  �  �  w  o  d  S  :    �  �  �  V    �  ~  �  �  �  �  �  	  D  \  X  R  K  =  +    �  �  M  �  �  7  �  �  �  �  �  �  �  h  >    �  �  G  �  �  L  �  �  �  M  �  �  �  �  �  �  �  �  X    �  n    �  �  9  �  G  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  \  B  (  �  �    �  �  �  �  �  �  �  �  �  �  {  X  1    �  �  �  �    �  �  �  �  �  �  �  �  �  �  �    |  x  u  z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  h  \  L  K  H  B  6  (      �  �  �  �  ~  K  3  ,    �  s  �  �  �  �  �  �  �  x  c  I  +    �  �  �  a  1    �  �  �  ~  }  }  t  j  Z  G  2      �  �  �  �  �  s  {    Y  4  <  C  H  F  <  .  $        �  �  �  �  d    �  B    3  �  �  �  �  �  �  �  �  x  ]  A  $    �  �  �  x  H   �   �  �  �  �  w  E     �  �  �  X    �  B  �  G  �  $  �  �   �  X  X  m  �  �  �  �  �  �  m  T  8    �  �  �  �  �  s  d  �  �  �      	  �  �  �  �  �  �  �  }  f  N  ;  -    �  �  �  �  ~  f  H  "  �    (    �  �  �  t  >    �  P  �          �  �  �  �  �  �  �  �  �  �  �  s  L  &   �   �      :  :  =  =  ;  2  "    �  �  �  �  O    �  �  R    U  ]  W  F  *    �  �  �  �  ]  *  �  �  a    �  n     �  O  O  N  L  F  ?  /       �  �  w  L  "  �  �  �  �  i  D  x  u  r  o  h  `  W  M  @  *    �  �  �  �  q  Q  >  %    �  �  u  `  K  6     
  �  �  �  �  �  �  �  u  [  A  (    +  +  ,  ,  -  .  .  /  +  "        �  �  �  �  �  k  F  [  H  5  "    �  �  �  �  �  �  �  }  m  ]  N  ;  (      �  �  �  �  �  �  �  �  �  p  <     �  X  �  �  1  �  �  2  �  �  �  �  �  h  P  6       �  �  �  �  d  C  !     �   �  �  �  �  �  �  �  �  �  �  �  �  �  w  ]  E  �  �  \     �  �  �  �  �  �  �  �  �  c  A    �  �  �  T    �  m   �   k  	>  	N  	;  	,  	'  	  �  �  �  �  U    �  �  l  8  �  �  �  q  	  �  �  �  �  �  w  R  ,    �  �  �  x  Q  $  �  �  r    �  �  �  �  �  �  ~  r  d  U  <    �  �  �  �  �  g  J  -  e  ]  T  L  C  :  ,    	  �  �  �  �  �  x  a  J    �  �  s  u  r  n  e  P  0  
  �  �  �  �  z  X    �  g  �  Y  �      +      #  #      �  �  {  F    �  ;  �  �  �  B  7  4  1  ,  &           �  �  �  �  �  r  U  7    �  �  D  5  %    �  �  �  �  ^  -  �  �  �  I    �  �  V     �  �  u  f  S  :    �  �  �  �  Q    �  �  l  ,  �  �  ]  �  L  G  ;  '    �  �  �  �  |  e  P  H  >  +  "    &  -  2  �  �  �  �  �  �  �  �  �  �  }  s  h  ^  S  I  ?  4  *    �  �  �  �  _  <    �  �  �  q  ?    �  �  l  4  �  �  �