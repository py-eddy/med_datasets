CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�n��O�<       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�e   max       PƯ�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���w   max       <���       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�33333   max       @F�G�z�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @v|�����     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q            �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @��            6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��G�   max       <���       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B u�   max       B4�       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B ?�   max       B4��       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�97   max       C���       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C��G       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          I       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          O       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�e   max       PƯ�       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u&   max       ?���#��x       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���w   max       <���       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�p��
>   max       @F�\(�     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v|�����     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q            �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @��            Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?t��Z��   max       ?���#��x     @  [L            	   	                                       +      )   
            
                                 +         	      $   "   H      
      $         	   #               
         
         
         
         NK�vOr�`Oa0�NEʯNPwN�eONpMOAoO���NV�Nyv�O m�O7=.P*�O?��POBO��rP���O�,P2��O�LN��;NY�N|N:O�NsE7P�NS/�NE�N���O�X,O/�Nm*$N�_�N���PA�Ok۩N�,OU�qO���O�b�O���PƯ�N��#O��O�gP!�?N��N���N��|O�R�N�|N�O���N��bN�IO�oN���N�&�N�2[O3�IN�	gNM�N�NN�}]N�PKNT\N�G�<���<�j<�1;�`B;ě�;��
;�o:�o�o�o�t��e`B�e`B�e`B��C���C���t���t���9X��9X��j�ě��ě����ͼ���������`B��`B��`B�����������\)�\)�t��t���P�����������#�
�#�
�0 Ž0 ŽH�9�L�ͽL�ͽY��aG��m�h�q���y�#�}󶽁%��%��%�����������C���C���C���\)���w�����������������������������	)34+&$& 	[aenz����zna[[[[[[[[
#.'#
	#(��������������������]az�����zwnddppnj_V]���������������abnux{�{nbb[aaaaaaaa��������������������)5BLNYONCB5)\gtw���������xtrokf\��������������������)59ACB:5)
 )5B[hhp���hO�����������
#b������{`K<�����#'0<Ua_Y8/#
��{����������������z{���������������������������������������� �        ��������������������vz}��������������zvvCHU^adaaUIHDCCCCCCCCmz�������������wkbcm����������������������������������������KNP[_gttwutphg[NIHKKtz�������������zupqt����������������������������������������������������������������������������������� %$������N[gt���������tgZPKJN��������������������imz������������zmkkiCHUajnz����zaULCBEC������������������������������������������*(-<cjF.������������������������������������������������#*D<><8#������gt���������������eUg������ �������������())58ABFBBA5*)%#((((������������������������������������innz����������|zpnii$)6:;A6+)$$$$$$$$$$$ot���������������trofgju������������tpgf;<HMTQLIHF<91568;;;;chtx�������tslhfd`\c��������������������##0<CIIJIIF<0/####/08<INMMMKI<40/.////_gt�����������tpgc]_�������������������������������������������

�����������0<HUW[_UHC<600000000_ahnz|������zxqnkea_���������������������� (%������������������(�#����������t�g�N�B�7�I�N�g¦°¯¨¦���ֺȺ��˺ֺ�����!�-�:�-�!�������U�M�H�G�E�F�H�U�W�[�\�W�U�U�U�U�U�U�U�U����ùñíù���������������������������ź@�<�;�@�L�Y�\�Y�S�L�@�@�@�@�@�@�@�@�@�@�#���
������ �
��/�8�<�D�B�A�<�/�.�#������������������ ���������������àÓÍÇ�z�q�n�f�w�{ÇÓàìú����ùìà�������������ʼҼѼʼʼ��������������������������������ĿʿοſĿ����������������0�0�)�)�,�0�3�=�I�J�V�W�a�b�W�V�I�=�0�0���������ƿݿ��������������ݿѿ��!�&�9�H�Z�g�s�����������������s�g�A�5�!���s�m�k�l�q�s�������������������������������m�T�4�"����"�T�y�����������������ʾž����ʾ����	��&�.�3�0�.�����׾����������|�I�5�N�Z�s��������������������®¦§²¿�����������
�����������¿®�ݿ����y�k�]�W�`�y�����Ŀ����"�!����ݾf�b�Z�S�S�T�Z�f�s���������������s�f�f�H�F�A�B�H�U�U�a�n�n�q�n�k�a�_�U�H�H�H�H�G�C�=�G�T�`�j�m�p�m�`�T�G�G�G�G�G�G�G�G������Ƽ�����������������������������������������ݿڿ׿ݿ�����������A�;�8�A�G�M�Z�Z�f�Z�Y�M�A�A�A�A�A�A�A�Aƞƌƀ�z�yƁƧƳ�������������������ƞ��������������$�$�$�������������������������������������������������
����������������)�*�4�0�)����������������������
��"�.��
����������6�4�-�)�)�'�)�*�6�B�O�R�[�e�e�[�O�G�B�6���������ûлػܻ��ܻлû����������������������������������ľþ����������������U�P�P�U�\�b�n�{��{�t�n�c�b�U�U�U�U�U�U�f�Y�L�7�5�@�M�Y�f�������ʼּʼ�������f��پ޾߾�������	�
��� ���	������������߾����	�	���	���������������������������	�����"�"�(�%���	�������������������Ŀѿݿ����ݿѿĿ����⺽�����������ͺ����F�[�[�F�!��������������5�M�g�Z�X�N�G�5����������.�������!�S�y���н��$�'���齷���M�J�@�7�>�@�D�M�X�Y�f�h�o�j�f�Y�M�M�M�M�Y�O�S�R�Q�Y�Z�e�r�{�~���������~�r�e�]�Y������ܻλлܻ�����'�3�:�@�M�@�4�żšŇ�{�r�_�vŇŦŸŹ���������������żŭũŠŠŠšŭŹżźŹŴŭŭŭŭŭŭŭŭ�h�g�h�o�tāĂčĚĚĚėčČā�t�h�h�h�h������'�(�4�=�@�G�M�M�M�@�4�'����c�r����������"�������ּ�����s�c�����������������������ʾ˾оʾɾ�����������������������������������������������¦²¿��������������¿¦���������������������������
�����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������������������������������ā�{�t�h�e�d�h�tāčĚĜĦįĦĢĚčāā�����������������Ľʽн׽ѽнĽĽ��������������ݽ�������&������������ĳĬĩĨįĶĻĿ��������������������Ŀĳ�=�;�0�*�0�4�7�=�I�V�Y�`�]�V�I�H�=�=�=�=ĿĽĿ������������������ĿĿĿĿĿĿĿĿD�D�D�D�D�D�EEEEEEED�D�D�D�D�D�D�FFFFFF$F1F<F4F1F+F$FFFFFFFF�������������¹ùϹܹ������ֹܹϹ������/�(�%�/�<�H�H�O�H�<�/�/�/�/�/�/�/�/�/�/ààÓÏËÇÓàìùùû��þùìàààà , m j � \ J " [ V N j 4 � j 0 C d g P Z / F 9 C \ 3 C 4 k F H 0 h : ^ I C O A 7 � _ _ 8 , ` 7 l Y p j ( A 4 _ d \ [ $ K J ' = ' Z s : @  X  @  �  �  u  ,  �  �  }  q  �  f    �  �  �  _  �  �  �  2  �  i  |  W  �  �  o  k    �  y  �  �  �  �  �  �  �    �  �  �  �  S    �  ^  �  �    �  D  1  f  �  [      �  �  �  q  �  �  ;  V  <���;ě���o�o�D��:�o�T���#�
���T���T����`B���ͽ49X��9X�0 Ž�P�y�#�H�9�}�+�t��������t���w�m�h�o���49X�@��aG��t���P�#�
���-�ixս,1�<j�u���P��hs��G��]/�H�9�}󶽡���@��e`B�m�h��{�u��7L���㽉7L��\)��t����-�����\)���㽙����C���{���w���
��1��1B��B�6B��BBUBB��ByB�uBnrB(?�B;!B�B
TB�Bf-B�jB"4B&flBB*Y�B"�B!~B.�`B�B ��B�TB B��B�B�4B �,B`�B_�B4�BWjBd9B	��B��B u�Bq�B<�B�hBA�B �B!F8B$h�B
��B�$B4�B)�B-Q~B�B�/B
��B
^B�BI�B��B%��B&o�B
.�BBp�BKB'aB%VB
��B��B�1Br[B��B��B�"BOB�BIdB�6B'�5B;.B�.B
>CBA5B�BERBnwB&�B9�B*:pB"NB!��B.�B��B �-B��B ?�B��B��B	�B? BH;B=.B4��B��B;(B	��B��B ALBB�BA<B��B�'B �yB!}�B$C/B<AB��B=B)��B,�<B@�B��BKDB
|'B�kBB?B��B%��B&KSB	ÑB��B�B@�B@�B@rB
�`BaA��?A��e@FR&A�|A�,�?�3�A�R�Aє�A�XT@��}AvD�B	�A~
�A��A�O�AkV�AW�
A��xA��A}`�AB��A��.Ag��B�A�ɏA<�B>B��A�؄A��A��Aأ2@�%AL�'A���@���AYy�AX��A���Az&�@P�8A�-2A��@�Xg?�x@���A���A��-A�q@��h@���AM'�A�^xA��A��4C���A���A��BA$�|A0��A��B�A��+C�KdC���>�97A�R�A��KA��{A��@I�qA��À�?Ɗ�A�nZA�abA���@� �AwidB?�A��A��A���Ak)�AW	A�{�A�uA~|�AB�A�RAe�_Bt�A��A<�~BS'B	:A��>A�jA�w7A؄�@�Y�ALs<A��w@�Y�AZ�QAX�A��Azެ@Kp�A�OPA�@���?�t@��hA��
A��A݆�@��sA�AK�A�~�A��A���C��A���A�3]A#?A2��A��B>tA��C�H�C��G>���A�0A��            
   
                                       ,      )                                                +         
      %   "   I      
      %         	   $            	   
                                                                              +      ;      =   #   3                     %                           +               +   #   O         )   /            3                                                                                             )      #      =                                                      +               +      O            !            +                                                   NK�vOr�`N�.;N�CNPwN�eONpMOAoOZUqNV�Nyv�N��O7=.P �uO?��O�w�O��P���O'l!Od��O�LN��;NY�N|N:N��NsE7O�P�NS/�NE�N���O�X,O/�Nm*$N]N�N���PA�N���N�,O5��O7��O�b�O�PƯ�N��#O��OLzO���N��N���N��|O�fN�|N�O���N��bN�IO�oN���N�&�N�2[O��N�	gNM�N�NN�}]N�C+NT\N�G�  �  >  �  g      �  �  �  �  P    _  �  �    �  K  [  �  �  �  &  z    �  �    �  �  �  �  �  	  y  �    �  `  �  �  �      �  a  �  2  J  =  �  �  R      �  z  '       �    �  n    �  �  �<���<�j<T��;ě�;ě�;��
;�o:�o�o�o�t���o�e`B�u��C��������ͼ�t��o��w��j�ě��ě����ͼ�h�����C���`B��`B�+�������o�\)�\)�0 Žt����,1���H�9�����#�
�<j�]/�0 ŽH�9�L�ͽP�`�Y��aG��m�h�q���y�#�}󶽃o��%��%��+��������C���C���O߽�\)���w�����������������������������),/*)"
`ajnz����zna````````
#.'#
	#(��������������������]az�����zwnddppnj_V]�����������������abnux{�{nbb[aaaaaaaa��������������������!)5BCMKB@5)\gtw���������xtrokf\��������������������)59ACB:5)
*06BOY[diy}t[OB6)( *���
����������
#b������{`K<����
#*/=HQPH</#
������������������������������������������������������������ �        ���������������������������������{������CHU^adaaUIHDCCCCCCCCmz������������}oifgm����������������������������������������LNRZ[gttnge[NNLLLLLLtz�������������zupqt����������������������������������������������������������������������������������� %$������T[gt��������tgg[VSTT��������������������kmz������������zmlmkGKTaenyz���zwna^UPHG������������������������������������������*(-<cjF.������������������������������������������������
03661#
�����fmpt������������utgf������ �������������())58ABFBBA5*)%#((((�����������������������������������innz����������|zpnii$)6:;A6+)$$$$$$$$$$$ot���������������trofgju������������tpgf;<HMTQLIHF<91568;;;;chtx�������tslhfd`\c��������������������##0<CIIJIIF<0/####/08<INMMMKI<40/.////`gt�����������tqgc^`�������������������������������������������

�����������0<HUW[_UHC<600000000gnz{������zyrnligggg���������������������� (%������������������(�#����������t�g�N�B�7�I�N�g¦°¯¨¦�ֺԺɺȺɺҺֺ�����������ֺֺֺ��U�T�H�H�E�G�H�U�V�Z�[�U�U�U�U�U�U�U�U�U����ùñíù���������������������������ź@�<�;�@�L�Y�\�Y�S�L�@�@�@�@�@�@�@�@�@�@�#���
������ �
��/�8�<�D�B�A�<�/�.�#������������������ ���������������ÓÐÊÇÂ�y�x�zÇÓàìùüÿûùìàÓ�������������ʼҼѼʼʼ��������������������������������ĿʿοſĿ����������������=�5�0�,�,�0�0�=�I�S�V�_�`�V�S�I�=�=�=�=���������ƿݿ��������������ݿѿ��$�#�(�:�I�Z�g�s���������������s�g�A�5�$���s�m�k�l�q�s���������������������������T�;�2�+�/�;�G�T�`�y�����������������m�T�׾ҾʾǾʾ׾����	��$�"�!��	���������������|�I�5�N�Z�s��������������������¿¹³´¿����������������������������¿�пĿ����������Ŀѿݿ����������ݿоf�b�Z�S�S�T�Z�f�s���������������s�f�f�H�F�A�B�H�U�U�a�n�n�q�n�k�a�_�U�H�H�H�H�G�C�=�G�T�`�j�m�p�m�`�T�G�G�G�G�G�G�G�G������Ƽ�������������������������������̿��������� �������������������A�;�8�A�G�M�Z�Z�f�Z�Y�M�A�A�A�A�A�A�A�AƪƖƌƌƖƧƳ����������� ������������ƪ��������������$�$�$����������������������������������������������������������(�)�1�*�)�������������������������
��"�.��
����������6�4�-�)�)�'�)�*�6�B�O�R�[�e�e�[�O�G�B�6���������ûлػܻ��ܻлû������������������������������¾¾��������������������U�P�P�U�\�b�n�{��{�t�n�c�b�U�U�U�U�U�U�f�Y�L�7�5�@�M�Y�f�������ʼּʼ�������f����������	�	�������	������������߾����	�	���	�������������������������������	��� �"�'�#������Ŀ��������������Ŀѿݿ������ݿѿĺ⺽�����������ͺ����F�[�[�F�!������� ������(�5�5�A�H�G�A�;�5�(������.�������!�S�y���н��$�'���齷���M�J�@�7�>�@�D�M�X�Y�f�h�o�j�f�Y�M�M�M�M�Y�O�S�R�Q�Y�Z�e�r�{�~���������~�r�e�]�Y���������������"�'�-�/�0�'������ŹűśŋŁōŠŭ��������������������ŭũŠŠŠšŭŹżźŹŴŭŭŭŭŭŭŭŭ�h�g�h�o�tāĂčĚĚĚėčČā�t�h�h�h�h������'�(�4�=�@�G�M�M�M�@�4�'�������|�����������!��������ּ��������������������������ʾ˾оʾɾ�����������������������������������������������¦²¿��������������¿¦���������������������������
�����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������������������������������ā�|�t�h�f�e�h�tāčęĚĦĬĦĚėčāā�����������������Ľʽн׽ѽнĽĽ��������������ݽ�������&������������ĳĭĪĩĲĳĹĿ��������������������Ŀĳ�=�;�0�*�0�4�7�=�I�V�Y�`�]�V�I�H�=�=�=�=ĿĽĿ������������������ĿĿĿĿĿĿĿĿD�D�D�D�D�D�EEEEEEED�D�D�D�D�D�D�FFFFFF$F1F<F4F1F+F$FFFFFFFF���������ùùϹܹ����ܹԹϹù��������/�(�%�/�<�H�H�O�H�<�/�/�/�/�/�/�/�/�/�/ààÓÏËÇÓàìùùû��þùìàààà , m 3 { \ J " [ X N j " � h 0 7 f g [ / / F 9 C L 3 7 4 k P H 0 h 4 ^ I = O 8 + � 8 _ 8 , b $ l Y p [ ( A 4 _ d \ X $ K < ' = ' Z ^ : @  X  @  �  �  u  ,  �  �  �  q  �      �  �  �  �  �  �  �  2  �  i  |  �  �  �  o  k  �  �  y  �  x  �  �  6  �  �  z  �  C  �  �  S  �  �  ^  �  �  �  �  D  1  f  �  [  �    �  a  �  q  �  �  �  V    >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  �  �  �  �  �  �  w  b  I  /    �  �  �  �  �  �  >  >  <  9  4  +    	  �  �  �  �  �  d    �  u    �  �  �    .  x  �  �  �  �  �  �  �  u  M     �  �  Y    �  v  8  N  d  l  r  s  r  n  h  a  X  ,  �  �  n  M  )    �  �    %  5  .  $          �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �    !  B  b  }  �  �  �  �  �  �  �  �  �  �  �  x  _  A    �  �  u  5  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  _  <    �  c    M  x  �  �  �  �  |  ]  2  �  �  x  /  �  �  /  �  2    �  z  p  g  ^  T  K  B  8  .  $                  $  P  @  0          �  �  �  �  �  �  �  {  i  S  ;  #     �  �  �    �  �  �  �  �  �  �  k  I  #  �  �  �  X  
  �  �  _  L  8  "    �  �  �  �  �  �  �  �  f  B    �  �  `  '  �  �  �  �  k  _  A  *        �  �  �  �  �  �  g    �  �  �  �  �  ~  x  q  k  c  Z  Q  H  <  .         �   �   �  �  	  �  �  �          �  �  �  ~  R  6    �  q  >   �  Z  m    �  �  �  �  �  �  �  �  x  H    �  �  F  �  8  �  K  %  �  �  �  W  &  �  �  �  F    �  �  w  W  =    �  0  D  >  6  *     0  L  [  U  L  :  $    �  �  }    `  �  �  �  �  �    D  �  �  �  �  �  �  �  V    �  g    �  �   �  �  �  �  �  �  �  �  �  �  �  n  M  &  �  �  �  �  �  w  n  �  �  �  �  q  X  8    �  �  �  g  ;    �  �  u  g  .  �  &  %  #  !                       #  &  )  ,  /  2  z  v  q  m  h  ^  T  J  ?  0  !    �  �  �  �  s  P  -  
  �  �  �  �  �    	  �  �  �  �  �  �  s  5  �  �     �   3  �  �  �  �  �  �  �  v  [  <    �  �  i  4    �  �  n  <  �  �  �  �  �  �  �  �  �  �  �  Z    �    +  �  u           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    u  j  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  _  D  (  	  �  �  �  y  X  �  �  �  �  �  z  j  W  A  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  W  7    �  �  P    �  p  M  .  �  �  �  �  �  �  w  i  Z  J  :  *      �  �  �  �  �  �            �  �  �  �  �  �  �  �  �  �  m  Q  "   �   �  y  o  e  [  O  >  ,    	  �  �  �  �  �  �  �  _  .  �  �  �  �  n  X  ;    �  �  �  M  �  �  X    �  �  m  .  t  f  �  �  �                �  �  �  v  B    �  �  @  �  �  �  �  �  �  �  �  u  c  R  C  7  +      �  �  �  �  q  ]  ^  `  M  8    �  �  �  �  �  �  �    n  ]  P  C  3  $  �  �  �  �  �  �  �  �  �  �  y  \  7    �  �  r  Y  B  +  �  �  �  |  m  K  :  \  Q  $  �  �  �  \    �  }  .  �  �  �  �  5  n  �  �  �  �  �  �  �  l  ?  
  �  V  �  N  �  v      �  �  j    �  �  �  �  �  �  r  J    �  G  �  �  ^    �  �  �  �  �  t  c  R  @  )    �  �  �  �  P  �  �  H  �  �  }  l  [  M  ?  5  .  &        �  �  �  �  �  �  �  Z  \  S  V  _  a  _  L  1    �  �  �  �  �  �  �  D     �  d  n  y  �  �  �  �  �  �  �  h  L  *  �  �  �  >  �  <    2  "      �  �  �  �  �  �  �    p  a  R  B  1        �  J  D  >  8  1  +  !      �  �  �  }  N    �  �  o  2   �  =  ,      �  �  �  �  �  �  s  _  O  S  W  N  ?  -     �  a  �  �  m  H  '    �  �  �  f  8    �  �  :  �  ?  �   �  �  �  �  �  �  �  �  }  z  y  x  w  t  q  �  �  �        R  @  .           �  �  �  �  �  �  �  �  �  M    �  �    �  �  �  r  7  �  �  �  Q  %  �  �  �  x  D    �  �  �            �  �  �  �  �  �  �  �    d  A    �  �  �  �  �  �  �  q  Y  A  (    �  �  �  �  �  �    $  8  J  \  z  d  M  6      �  �  �  �  �  �  o  Q  /    �  �  =   �  !  %  $      �  �  �  |  L    �  �  <  �  �  W    �  H     �  �  �  �  �  �  �  �  �  {  j  J  !  �  �  j     �   Q        �  �  �  �  �  �  �  �  n  ^  N  A  8  /    �  �  l  {  �  u  i  ^  Y  U  E  1    �  �  �  d  %  �  �  v  9    �  �  �  �  �  �      	  �  �  �  �  �  �  m  N  ?  2  �  �  �  �  �  �  �  �  �  �  �  �  y  q  h  ^  U  K  B  8  n  R  5    �  �  �  �  ]  0  �  �  x  1  �  �  ;  �  |      �  �  �  �  e  C     �  �  �  �  �  �  o  T  6    �  �  Y  �  �  p  Q  1    �  �  �  �  f  0  �  �  y  6  �  �  m  �  �  �  �  �  �  �  �  �  o  Z  B    �  �  �  !  �  K  �  �  �  �  �  �  �  �  �  s  d  U  D  3      �  �  �  �  �