CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�Q��R       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P���       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��;d   max       <�1       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?aG�z�   max       @FC�
=p�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v33334     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P            �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�h        max       @�|�           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �n�   max       <T��       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��D   max       B0z7       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B0@�       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >l��   max       C�ƾ       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >I,x   max       C���       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          _       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P�f       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��=�K^   max       ?�!�.H�       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��S�   max       <��
       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?aG�z�   max       @F0��
=q     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v}\(�     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P            �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�h        max       @�4�           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A   max         A       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?p�)^�	   max       ?��rGE9     �  \8                         6                                 1            %                              	      #                        #                  _      1      
   	      (                     )         >         O;�tO�N~��N�A�NGh�P^ЎN㗬P*9�N�3�OG5N�tjP21^O���O�%N��N�$$Or��OZЎP���O���NC��O�0�O���O($N��_NsL�OkeN�E�O&"N�*N" dN��2N�YNX�P"�Ou��Np`�NY/�N�O\)NE�yNA�eO�9~NM֨O�1qN-@N���M��PybN��9O��N��KNN�nDNeO�/�N�r	O�O��Nt��N�8�OɭO���N�Oh�vOxÃOZ�:N�8N��<�1<�t�<�o<49X;ě�;o;o:�o��o�o�o�t��D���D���T���e`B�e`B��C���C���t����
�ě��ě����ͼ��ͼ�/��`B��`B��h��h���������+�C��C��C��t���P��P��P������w�0 Ž49X�49X�8Q�D���L�ͽY��e`B�}�}󶽁%��%��o��o��t���t�������P���w���w��\��;d��;d����������������������������������������cnz��������znmcccccc�)-1,)�����AHIUaddaUPHEAAAAAAAA'BK[��������}t[B5&'������������������������
'/9G@/#
������V[ht������tthf`[VVVV��)+*(#�����"#/5<AF><4/'#"""�������������������������������������������������������zy|~�����������������������������������������"/6COX\_ed\VC6*$>BLNYgty|vtgc[NF98=>���	#<b{�����fUI0�������������������������������������������)7O[htv�}oh[OB6)$")Vet������������h[VTV������������������������������)*))),48985) ����NN[egmhg[NDCNNNNNNNN����������������������������������������,0<INNI<0+,,,,,,,,,,EHT]abkgaaTLHBCCEEEE
#/7<AA></+#

��������������������������)#���������y|�����������������y����������������������������������������KOW[chrmh_[ODEKKKKKK�����������������������������������������������������������
"0)������}���������xw}}}}}}}})3EN[t�������t[B)!��������������������������

����������������)BINg~�tgN5��-/37<EDE@=<2/'+,----�����	)"������������� #%�����35:BNOU[\__[NBA:5433#026500*#!*016<>E<0.*)********djt������������th^\d��������������������FHLU_ahnu|zpnaZUMIHFz{��������������{yyzCHJU^_aacaUTJHCCCCCC\amnswz�������}ztna\;<<>FHUY^cgmcaULHC<;Xbt�����������tga[WX����������������������������������������UQH</$
�����
#/<HU��
��������(/<@FHIJJIHH<5/+%#((NUabnqxvna]USONNNNNN�s�n�g�e�e�g�s�������������������������s�����������������������
���� ��
�������������������������������������������Һɺź��������úɺֺ�������ߺֺɺɺɺ�����������������������������������������ѿӿĿ�����������5�I�H�5�)������ѻ!������!�!�-�:�B�F�O�R�F�:�9�-�!�!���s�g�]�X�Z�g��������������������������/�/�+�%�)�/�<�H�U�V�U�U�I�H�<�1�/�/�/�/�s�f�]�Z�T�O�T�f�s�������������������s�<�:�6�<�@�H�U�a�e�n�y�z��z�n�a�U�H�<�<���y�`�N�F�C�G�`�y�����ȿԿ�����꿸���J�#����"�;�T�m������������~�x�g�a�J������j�s����������Ⱦվ���
����׾����������(�4�=�A�H�M�N�M�D�A�4�(�����������������ĿѿѿҿѿĿ���������������׾ʾž¾ɾ׾������	����������������������������)�,�)�&�������������������q�d�J�B�I�Z������������������Ŀ������������v�r�v���������Ŀҿؿ߿޿���
�����$�+�0�0�0�$�#��������_�g�n�s�v�r�s�����������������������s�_�Y�@�)���!�,�X�e�r�~���������~�y�r�e�Y�t�h�[�U�O�K�O�T�Z�[�h�tāćčĔęčā�t�������y�m�g�j�m�y�y�z���������������������������������Ŀѿ׿ѿпĿ����������������������������������������������������������������������������������������������4�(�����&�'�4�@�M�Y�Y�Y�Z�]�Y�M�@�4�лʻƻллܻ�����
������ܻлллм@�?�<�?�@�M�X�W�N�M�@�@�@�@�@�@�@�@�@�@�5�0�*�(�(�(�5�A�N�O�V�Z�_�Z�N�A�5�5�5�5�N�B�6�2�-�3�6�9�B�O�[�_�_�[�U�[�^�[�O�N��������(�,�(������������a�f�w�����������!�%����］�����r�a������������������������������ ���Ɓ�}ƁƃƎƚƧƲƳƸƳƧƚƎƁƁƁƁƁƁ��ھھ���� � ��������������ù����������ùϹֹܹ߹߹ܹϹùùùùù����������������
�#�-�0�<�F�<�0�'�	�������U�K�S�U�b�n�{ŀ�{�u�n�b�U�U�U�U�U�U�U�UÇÁ�z�z�zÄÇÓÕÞÜÓÇÇÇÇÇÇÇÇ�h�[�O�I�O�V�[�tčĦĮĭĪĦĝĐčā�t�h�	��	�����"�&�)�"��	�	�	�	�	�	�	�	���׾ʾƾǾԾؾ���	�������	����׾Ҿ׾ھ�����������׾׾׾׾׾׾׾׻����������������!�#�!�����ɺǺºȺɺʺҺֺ���ֺɺɺɺɺɺɺɺ�Ç�t�U�H�F�U�a�uáìù��������������àÇD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��M�@�6�6�8�@�M�g�r�����������������f�M�z�w�m�m�f�a�_�^�^�a�m�z�z���������{�z�z����
����(�*�6�C�G�J�D�C�6�.�*�������������������ĽĽнؽнĽ��������������������� ���������������������#��
����!�/�H�U�a�n�r�s�e�R�H�<�/�#�'��'�/�3�>�@�L�Y�e�k�e�^�Y�L�G�@�3�'�'�������������������Ŀѿݿ��ݿٿѿĿ����û������������ûлػܻ��� ������ܻлþ���������	�����	��������������׾־ʾʾ��������������ʾϾپ׾Ծ׾׾׾��������z�m�e�d�m�z����������������������ĿĳĦĜęĚĦĿ����������������������Ŀ�<�;�0�-�#��#�0�<�C�I�R�U�Y�U�I�<�<�<�<ĿľĳčĉčęĦĳ�����
��
����������ĿEE*E6E5E8ECEKELECE7E*EEEEEEEEE�S�N�N�[�`�l�y���������ýĽǽĽ������`�SFFFFFF F$F1F=FJFKFJFCF=F9F1F$FFF�����������ùϹܹ��ܹӹϹù�����������  E r > N P # 5 P 1 > M @ ` # u $ V [ O \ > 4 6 2 6  1 3 Y ] D @ H n n d   @ h Z 0 S F L l < � 4  ) } @ @ r F l B L T ` ? < j u b m P T    �  %  �    p  �  �    �  �    p  *      �  �    �  a  �  �  !  o  �  �  �  �  h  �  c  �    �  �  9  �  d  �  ?  t  V  W  i  '  P  �  Q    �  �  �    �  `  E  �  T  F  �  �  D  q  �  �  &  S    �;D��;ě�<T����`B;D����h�ě��Y��49X��j�������t��C����ͼ�t��\)��h�����P���ͽP�`�y�#���+���0 Ž\)�,1�#�
�\)�t���w��P��7L�D����P��P�T���Y��,1�0 Ž�t��'�%�@��P�`�@��V�q��������+��+��\)��7L���`��\)��j��t����-���w���
����ȴ9�n���\)����B�Bk.BH�Bn�B�zB
0�B,��B�B0uB=�B'B*�By�B��B ��BLLB0z7B��B%��B��B RB�,B�B2�B��B��B�B��B"+B"B&?cA��DB�`BOB-L�BL5B�YBmtBBB�B�gB?3B�lB�B��BD�B"��B#��Bz�B�gB�B��B�B%Y�B%��B
i�B=�B(kB)�#B�B��B�LB
?)B��BB��B��B�FB7�Bt@B��Bg�B�B��B
�B,��B5LB@�BjB.IB*�BM�B�B!*&BM�B0@�B�*B%r�B�BH_B�jB��BC�B?�B��B�vBĎB!�fB"3�B&D\A���B�B�;B-�B ��B��B�FBB�B��B�B?0B=fB�mB	�B>zB"�B#�B;�B��BƭB+�BȞB%>�B%��B
��BZB@kB)�MB��B�B��B
=�B�4B��B�uB��B?iBClA��A��KA���@;A�iA��@v[A�=�AÍ�ADY�A�~�Arh�A�pAP�A7�+Av��AV�A��A��2At�B	_yA���?�w'A��XAnf�AwߘA�nA���@�[C@�0@�V�A��VA���A�ɫ@��]B�B9�AV��>��pA��A���A�^A�VA��aAX>�AU�r@[BZ@;�2A�C��Y@ތ�A��A���A$��A2A��	?�n"Aw�@��dAY�{AO8�A��A��A�A㏟C���A4�C�ƾ>l��A�}�A�_�A�v�@;�A�"�A�[@wJ�A��AûEAD=SA�z8Ar�
A��5AMJiA7�Ax�AV��A�M�A��AuvRB	��A���?�ƦA�s�AmAw1!A���A��|@� �@��@ոA�
~A؇A�|�A�B��B7+AV�>I,xA��hA�{XA�yIA܆IA���AYuAUC@Zf�@3�JA��C��@��A���A�A#��A2�eA�(?��KAy�@��AYAO*�A�,YA�+A���A�~�C���A,dC���>��`                      	   7                                 1            &      	                        	      #                        #                  _      2         	      (         	            *         >                            5      +            5   %   '               ?         !   #                                    5                              !            7                                                                                                )                     =            !                                    1                                          %                                                            O)��O�N~��N�A�NGh�Oc%�N㗬O���N�Z�OlnN�hP �N�9OD{N��aN�$$O��OZЎP�fO@�eNC��O���O��N���N��_NsL�OFE_N�E�O&"N�ɹN" dN��2N�YNX�P��O]�/Np`�NY/�No�,O3�!NE�yNA�eO�9~NM֨O�jN-@N���M��P
�N��9Og��N�	NN�nDNeO;lN�N�O�O��Nt��N`7�OɭO��N�Oh�vN��fOK�cN�,�N��  �  ,      y  �    �  i  a    �  �  �  �    M  �  �  �  -  5  o  �    �  �  �  �  �  �  B  �  �  S  �    �  �  L     �  �    $  �    �  	E  G  �  �  �  �  R  �    �  c  K  +  D  �  �  �  �  �  
r  �<��
<�t�<�o<49X;ě��u;o��t��o�ě��e`B�u���ͼ�t��e`B�e`B��1��C���t���j���
������h�����ͼ�/����`B��h�����������C��\)�C��C�����w��P��P�����,1�0 Ž49X�49X��t��D������]/�e`B�}�}󶽙����o��o��o��t����������w���w���w��;d�ě���S���;d����������������������������������������cnz��������znmcccccc�)-1,)�����AHIUaddaUPHEAAAAAAAA^gt�����������xjg`[^������������������������
#*010'#
�����_ht������~thhb______�&'$  �����#/;<@<8/#!��������������������������������������������������������������������������������������������������'*.6CCOPUVROEC6*&#$'>BLNYgty|vtgc[NF98=>���
#<b{�����{UI0�������������������������������������������$)6B[ht{~zlh[OB6+&$$Zht������������h[YVZ������������������������������)*))�))1561)����NN[egmhg[NDCNNNNNNNN����������������������������������������,0<INNI<0+,,,,,,,,,,EHT]abkgaaTLHBCCEEEE
#/7<AA></+#

��������������������������$'!��������z}����������������zz����������������������������������������LOY[_hokh[OGLLLLLLLL�����������������������������������������������������������
"0)������}���������xw}}}}}}}}&/5N[ot������t[N6)'&��������������������������

���������������)5>HNTQB)����-/37<EDE@=<2/'+,----�����������������"$������35:BNOU[\__[NBA:5433#026500*#!*016<>E<0.*)********rt�������������}tkkr��������������������FHLU_ahnu|zpnaZUMIHFz{��������������{yyzCHJU^_aacaUTJHCCCCCCxz�������{ztxxxxxxxx;<<>FHUY^cgmcaULHC<;[dw�����������tgd\Y[������������������������������������������
###!
��������	��������)/<?EHHIHH<7/+&$))))NUabnqxvna]USONNNNNN�s�p�h�f�f�g�s�������������������������s�����������������������
���� ��
�������������������������������������������Һɺź��������úɺֺ�������ߺֺɺɺɺ�����������������������������������������ݿѿ˿ȿ̿ѿݿ���������� ����ݻ!������!�!�-�:�B�F�O�R�F�:�9�-�!�!���������t�u�����������������������������/�,�&�+�/�<�H�U�U�U�T�H�G�<�/�/�/�/�/�/�f�a�[�Z�X�Z�f�s�����������������s�f�f�H�A�?�H�J�U�\�a�k�j�a�U�H�H�H�H�H�H�H�H�y�`�U�M�M�T�`�y�������Ŀοݿ��ѿ����y�H�B�;�/�+�,�/�;�H�T�`�a�a�a�U�T�H�H�H�H�ʾ������������������˾׾��������׾ʾ������(�4�<�A�G�L�A�@�4�(�������������������ĿѿѿҿѿĿ���������������ݾ׾Ͼʾʾվ׾������	�
���	������������������������)�,�)�&�������������������r�f�L�D�L�g������������������Ŀ������������}�������������Ŀ̿ҿѿƿ���
�����$�+�0�0�0�$�#��������s�g�i�l�v�y�v�������������������������s�Y�B�/�%�"�'�3�@�O�Y�r�~�������{�u�r�e�Y�[�[�R�[�^�h�t�~Āāćā�t�h�[�[�[�[�[�[�������y�m�g�j�m�y�y�z���������������������������������Ŀѿ׿ѿпĿ����������������������������������������������������������������������������������������������4�(�����&�'�4�@�M�Y�Y�Y�Z�]�Y�M�@�4�л̻Ȼлһܻ�����������ܻлллм@�?�<�?�@�M�X�W�N�M�@�@�@�@�@�@�@�@�@�@�5�0�*�(�(�(�5�A�N�O�V�Z�_�Z�N�A�5�5�5�5�N�B�6�2�-�3�6�9�B�O�[�_�_�[�U�[�^�[�O�N��������(�,�(������������r�i�x������������!�%����］�����r��������������������������������Ɓ�}ƁƃƎƚƧƲƳƸƳƧƚƎƁƁƁƁƁƁ��ھھ���� � ��������������ù����������ùϹԹ۹ܹϹùùùùùùù������������������
��#�)�/�#��
��������U�K�S�U�b�n�{ŀ�{�u�n�b�U�U�U�U�U�U�U�UÇÁ�z�z�zÄÇÓÕÞÜÓÇÇÇÇÇÇÇÇ�h�[�O�I�O�V�[�tčĦĮĭĪĦĝĐčā�t�h�	��	�����"�&�)�"��	�	�	�	�	�	�	�	���ܾ˾;׾ؾ޾����	�����	� ����׾Ҿ׾ھ�����������׾׾׾׾׾׾׾׻����������������!�#�!�����ɺǺºȺɺʺҺֺ���ֺɺɺɺɺɺɺɺ�Ç�n�c�V�V�`�nÇåìù����������ìàÓÇD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��f�Y�M�@�?�A�H�M�Y�f�r��������������r�f�z�p�m�g�a�`�_�_�a�m�y�z�������~�z�z�z�z����
����(�*�6�C�G�J�D�C�6�.�*�������������������ĽĽнؽнĽ��������������������� ���������������������#� �� �#�/�3�<�H�U�Z�a�j�a�Y�U�H�<�/�#�3�(�0�3�?�@�L�Y�e�j�e�]�Y�L�E�@�3�3�3�3�������������������Ŀѿݿ��ݿٿѿĿ����û������������ûлػܻ��� ������ܻлþ���������	�����	��������������������������ʾ;־ʾʾ������������������������z�m�e�d�m�z����������������������ĿĳĦĢĞĜĦĳĿ��������������������Ŀ�<�;�0�-�#��#�0�<�C�I�R�U�Y�U�I�<�<�<�<ĿľĳčĉčęĦĳ�����
��
����������ĿE*E&EEEEE*E*E,E7ECEDEEECE7E7E*E*E*E*�S�O�O�\�`�l�y���������������������l�`�SFFFFFF$F$F1F=F>FBF=F8F1F$FFFFF�����������ùϹܹ��ܹӹϹù�����������  E r > N 2 # 1 = * ( ; / c ' u ' V Z D \ = / + 2 6  1 3 U ] D @ H l q d   < \ Z 0 S F M l < �     r @ @ r ? l B L T . ? 6 j u % h > T    h  %  �    p  �  �  s  �  >  �  m  �  �  �  �  *    m  �  �  y  �  �  �  �  �  �  h  �  c  �    �  �    �  d  u  �  t  V  W  i  �  P  �  Q  c  �  �  F    �  `  Y  �  T  F  �  p  D  #  �  �  �    �  �  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  r  �  �    }  {  u  m  `  P  >  ,      �  �  �  w  L  8  ,  &  !        �  �  �  �  �  �  �  r  T  /  �  U  �  J        
       �  �  �  �  �  �  �  �  �  �  l  T  <  $        �  �  �  �  �  �  f  ?    �  �  Y  �  �    �    y  u  r  n  k  g  c  \  S  J  A  7  .  '  $             �  �  �  �  �  �  �  ,  Y  w  �  �    d  >    �  �  (  X                �  �  �  �  �  �  �  �  �  �  �  �  w  %  G  Z  j  |  �  �  �  �  �  j  I    �  �  '  �  J  �    F  X  h  f  c  ^  U  E  3    	  �  �  �  �  o  8  �  �  u  #  C  V  ^  `  \  V  O  D  7  )      �  �  �  �  ~  _    �  �  �  �  �  �       �  �  �  �  �  �  i  C    �  �  �  �  �  �  �  �  �  �  �  �  l  L  "  �  �  �  P    �  @   �  �  �    %  6  @  H  ~  �  �  �  �  �  �  w  T  '  �  �  ~  U  _  s  �  �  �  �  �  �  x  g  T  ;    �  �  �  G  t   �  �  �  �  �  �  {  s  j  _  L  3    �  �  y  :  �  �  }  A      
     �  �  �  �  �  �  �  �  z  p  f  h  m  r  w  |  �    2  ?  F  K  M  K  F  =  0    �  �  �  g  .  �  �   �  �  �  �  �  y  h  W  D  0        �  �  �  �  �  v  /  �  �  �  �  �  �  _  #  �  �  h  =    �  �  4  �  �  N  �  6  �  �  �  �  �  �  �  �  �  �  �  �  x  b  G  %  �  �  e  �  -  %        �  �  �  �  �  �  �  �  �  p  ^  L  9  '    *  4  0  (      
    �  �  �  �  �  s  N  %  �  �  w  �  B  ^  l  m  d  Q  -  �  �  |  2  �  �  {  :  �  �  
  �   �  p  y  �  �  �  �  �  �  �  �  }  h  R  9  !    �  �  J  (              	    �  �  �  �  �  L     �   �   �   [   )  �  �  �  �  �  �  �  �  u  g  Z  O  C  8  ,        �   �   �  y  ~  �  }  v  k  ]  M  ;  )    �  �  �  s  ;  �  �  Y  ;  �  �  �  �  �  �  �  �  �  p  [  E  /      �  �  �  �  �  �  �  �  �  �  �  �  �  q  U  8      �  �  �  �  �  Z  1  �  �  �  �  �  �  �  �  �  �  �  �  �  x  W  +  �  �  X  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  o  g  ^  B  @  >  =  ;  9  8  6  0  %        �  �  �  �  �  �  �  �  �  �  �  �  �  f  A  (    �  �  �  �  W  !  �  �  a    �  �  �  |  h  T  A  )    �  �  �  �  �  h  K  6  F  W  g  K  K  :  0    �  �  �  �  a     �  �  �  c  /  �  C  �  e  �  �  �  �  v  x  n  c  }  g  ]  ]  -  �  �  y  /  �  i   �              
    �  �  �  �  �  �  �  �      (  7  �  �  �  �  �  �  �  �    y  t  n  i  ]  F  /       �   �  t  �  �  �  �  �  �  �  |  k  V  ?  *    �  �  k  /  �  �  -  +  B  B  0      �  �  �  �  m  M  H  D  1        7        �  �  �  �  �  �  �  �  �  �  �  M    �  r  $   �   �  �  �  �  �  �  �  z  f  M  4    �  �  �  �  r  P  +     �  �  �  x  L  B  #  �  �  �  `  $  �  �  Q  ,  �  )  N  Y  P    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      #      �  �  �  �  �  �  �  �  �  v  W    �  �   �  �  �  �  �  �  z  d  O  :  %    �  �  �  �  �  �  t  \  E    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  S  ,  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  K  0    �  3  �  �  	)  	A  	D  	.  	  �  �  d    �  �  /  �  �  �  �  G  :  '    �  �  �  �  �  �  �  o  4  �  �  �  P    �  �  �    C  m  �  �  �  �  e  =    �  �  x  ,  �    �  �    P  �  �  �  �  �  l  K  )    �  �  �  h  D    �  �  �  �  �  �  �  �  �  ~  j  T  <  $  	  �  �  �  �  x  `  I  b  �  �  �  �  �  �  �  �  �  }  j  U  ?  (    �  �  �  �  b  ;  R  H  >  4  *  !        �  �  �  �  �  v  ]  J  9  (    �  	  8  `  w  �  �  |  g  ?    �  s    �  I  �  .  �  �  �        �  �  �  �  �  �  �  �  q  Z  ?       �  ~  7  �  r  G    �  �  �  W  (  �  �  `  	  �  /  �  9  �    s  c  Y  O  D  9  .  "      �  �  �  �  �  i  A     �   �   �  K  F  A  =  :  B  J  Q  N  9  %    �  �  �  �  �  w  \  A  �  �    '  !    
  �  �  �  �  �  �  �  �  �  �  }  p  d  D  7  +      �  �  �  �  �  �  �  v  c  U  K  A  >  ?  @  �  �  �  �  �  q  >    �  �  &  �  K  �  9  �  =  �  �   �  �  �  �  �  �  �  x  l  _  S  G  ;  /  #    �  �  �  �  �  �  �  �  �  �  �  {  \  ;    �  �  ~  .  �  �  <  �  �  %  #  
�  
�  
�  
�  �    �  �  �  �  )  �  (  
�  
  	o  �  $  r  �  �  �  �  �  a  =    �  �  �  t  E    �  x    �  O  �  
  
a  
h  
O  
+  	�  	�  	  	9  �  �  O  �  �  S     �  O  �  �  �  s  X  <    �  �  �  �  l  B    �  �  �  j  ;    �  �