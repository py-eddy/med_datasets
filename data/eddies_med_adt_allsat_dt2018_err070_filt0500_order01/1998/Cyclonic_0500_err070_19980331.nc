CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��j~��#      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P՞D      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���T   max       <�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F\(��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @v~z�G�     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q�           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��           �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       <�t�      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B5�      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4�      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�HW   max       C�֯      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?���   max       C��      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Q      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          M      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�9U      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�n��O�<   max       ?�2�W���      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���T   max       <�h      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F\(��     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @v~z�G�     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @N@           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_ح�   max       ?��J�L�     �  V�               D                  F   >      #   /   L            9      /      #               #   :      Q   -      !                     2      	         
   3                           
               	   N���N6��NW��N�]�P՞DOV�~N�fO9� N1�O�ѠP�:@P�uO���O���Pu�PD8!O!�N:yOP��NWw�P�O\��P2�P%wO�7�Oe�NO�ؼO���Pt�Py��P!�iPQ��N5l�O��O;�'N&��N��P �mOQ}gN�GO�B�N���N
v�N�E�O=CyO1��P�4N�XM���O�R\OSˎN~{�NO@N���N\X&NY
N� �O<�8N�%�O��N���N��"<�<�9X<���<u<o�o�o�D���D���D����o�ě��ě���`B�o�t��#�
�#�
�49X�D���D���u��t���1��9X��j��j��j��j��j������������������`B��h�����+�+�t���w�'',1�8Q�8Q�<j�@��D���D���P�`�T���]/�aG��e`B�ixսm�h�y�#��o��+���T���T�� 	������������������������������������������������


#/<F<5/#




���#<U�����{I0
����
#3<BECB=</#������������������������	�������-0<IU^USI<20--------���6CJOV__YOC���8a���������|aTM;5528����
85��������������������������������������������������#)5MSSQQNB5-
	�����������������7;HTZ[ajdaTTH;62/277������

����������� ����(/5B[gz���|pg[A5) (zz����������|zzzzzzzO`n������������naQMO�����
#*,/+#
����o����������������xoo<N[^t������tVGAFFKB<;HTWagp{���zaTJ@:88;������������������������������������v{���������������zvv��������&62������/Unz����x`UH<,!)#����$ ����������)Bht��v[O)������������������������-05BN[gpospr]NB5$ #-�������
������������������������������������������������������,.'
�������5<@BO[ehie`[ZLB86515nqpu{�����������{ynngt�������������tg]\g��������������������ehtx}���tthdeeeeeeee����������������������������������������������������)Nt����������zN)	���������|}��������9<HIJJHD<;9999999999�����������������yz�"&0<IJJKI<80"
����

��������������������������!#+/8<HB=</&##!!!!!!#'*),,#QUajnqnaYUROQQQQQQQQ!! ��������������������[anz�������znda[[[[[LN[gt�������tg[NGIFL')55BNONGBA95)(&''''U[]gmt�����wtg[TUUUU²°¦¥¦©²¿��������������¿²²²²���������ĽнؽٽнĽ�������������������������������������������������ùøìããåêìïùÿ����������ùùùù�	�����g�P�@�<�(�.�Z��������	�� �'��	�)�'�%�%�)�/�6�B�O�[�h�j�k�^�^�[�O�B�6�)�����������������������Ⱦž��������������ʾ����������׾������	���	�����׾ʼ������������żʼҼҼʼ¼����������������;�.�"�	����ݾ����"�.�;�A�E�G�L�P�G�;����ֿɿɿѿ���5�s�����������C�5�(��������������8�a�z�����������H��������Z�M�4�)�,�4�?�D�M�Z�s��������������s�Z��������������������$�1�:�>�:�0�$��������s�d�Z�G�G�g���������������������������_�;�5�:�:�S�l�������û���λû����x�_�����������������	�������	��������ɺ��������������ɺֺ׺ֺԺ˺ɺɺɺɺɺɻ������������!�-�:�A�:�8�0�/�-�!��������z�q�w���������ѿݿ������ѿ�������������������������������������������s�m�X�Q�O�Q�Z�g�����������������������������׾ʾž��žʾ׾�������	��	���y�`�N�K�P�y�����Ŀѿ��ڿٿ�׿ǿ����y���������������־�	����	����ʾ�����ĿĦĖĦķ���������
��������������ĿÞÓÏÎÓàìù������������������ùìÞ����������������*�7�J�O�D�6�,������ɺ����r�Y�N�N�c�r�������ֺ��������ɼ��f�\�^�o�{�����ּ��!�.�8�9�!���ּ����D�D�7�'�%�*�;�T�������������������z�a�D��r�Y� ��&�4�]�r��������Ѽʼ�����������s�f�h�f�o�w�t�z�����������������������Z�N�S�Z�g�j�s�t�s�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z������������������*�6�O�Z�^�X�C�6��à×ÓÑÉÁ�{�ÇÑÕàáìñùûùìà�\�\�U�\�h�u�w�w�u�h�\�\�\�\�\�\�\�\�\�\�B�=�6�)�!�)�6�6�6�B�D�E�B�B�B�B�B�B�B�B�ѿĿ������������Ŀѿݿ�����������ݿѿ"��	���������	��.�D�G�T�^�T�N�G�;�"�@�4�'����
���'�.�4�7�@�K�M�N�M�@�@������������)�O�[�d�[�O�C�=�<�7�)���[�U�X�[�h�tāĊčĘčā�t�h�[�[�[�[�[�[�����'�3�3�4�4�3�,�'���������Y�O�M�@�4�'�%�'�4�?�@�M�P�Y�`�f�j�m�^�Y�.�$�!������������!�'�7�:�E�?�B�9�.�>�=�@�G�K�H�L�W�e�q�v�w�x�u�r�e�Y�L�B�>ĦĚď�v�m�lĀĦĪĿ�������
�����ĽĳĦ�������ʾ׾۾׾־ʾ���������������������F$FF#F$F1F=FDF=F6F1F$F$F$F$F$F$F$F$F$F$���x�g�a�^�]�_�i�x�}�~�������ûл��������лû����������������ûŻƻлܻ�������EEEEE'E*E7E?ECEPESE[EPECE7E5E*EEE�0�/�#�����#�*�0�6�<�@�<�0�0�0�0�0�0E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��ݽֽнĽ��Ľнݽ���������ݽݽݽݽݽ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��ݽӽ۽ݽ������������ݽݽݽݽݽ��0�,�$� ��)�0�=�I�V�b�o�t�o�k�b�V�I�=�0���������������ĿɿѿҿҿԿѿϿĿ�������������������������)�+�2�7�8�4�)�����a�^�U�U�O�T�U�U�a�l�n�z�}�{�z�n�a�a�a�a�<�3�/�'�#��!�#�/�<�F�H�I�L�M�H�<�<�<�< & ) J b j 9   W s S \ Q H N N K ! C C 2 h H T B R g % ) x b I F D 7 % l A M I Q / B \ Y Z N ` R = c g U � y ( d 3 ` A U ! W P    �  I  u    �  �  �  �  �  C  �    D  �  �  l  Y  ]  C  �  �  �  �  1  �  �  �  \  �  �  <    �  A  �  �  /  ;  �  �  �  �  �  J  �  �  �  �      �  �  �  �  �  �  h  �  �  7  &  �  <�t�<�o;ě�;D���q�������o�����`B�o���P��C������'aG���{��C���C���9X��C���C��}�C��e`B�T���L�ͽ��T���m�h���T�Y������\)���u�<j�o�<j��o�49X�,1��9X�P�`�H�9�Y��m�h�aG��Ƨ�aG��L�ͽ��㽍O߽�7L�m�h��7L�q����+��o��1���㽶E���E����BX�B*��B4\B��B%��BƩB5�B.�B&~�B/�A�_IB^�B!��B��B�B�>A���B#�	B�B��B�+B�B�B+%B	��A��lBYB1�Bx[B-)�B:}B.mB�%BX�B�B��B�4B�B��B��B(�yB
G�B`KB-�B xbBB ˉB�B;`BvB �B%�B�WBX�Bb�B%GRB(BnxB)�B'�B	Z(Bs;B	�ZBBB*�!B?�B�'B&�HB��B4�B?�B&��B/�A�\BB�B!�oB�B=�B��A���B#ǝB�AB��B̝BAnB��B*BTB	�8A��
B �cBl�BAXB-@aBT�B?�B�CBNB��B�B��BĎB�BG�B(�B	�?B��B9�B ��B��B ��BAZBBaB�B �QB%7bB��B?�BAGB%@�B;yB@BC�B�.B	2�B>cB	�.A��OA'q�A�$A���A��A��qALYAUa�@�_A]�MA��A�� A@�aB��A�X"@�+A�s@1��@i#At��A�A���AU��As7dAXxIA�ŎA�|cA��@:@��DA�?W@ݿGA�n;A�!%A�LQA�HBC�Aו*A{�\A`r@�A���A�z?�HW@�'!ARu?�	pA�TAM�C�֯@��@��IC���A�)C�1�A+�C��A.��B&�Aw�NA��A���A�7?A���A'�AќA�rAA�nuA؀HALrASg@���A]%FA���A���A@�~B	
�A�|�@�:�A��-@4�@di?At�A�c�A�zyAU3Ao&�AY%A�A��YA��i@3~�A�fA��i@��A�kA���A��(A���B�A���A{�Ab�3@�
�A�xGA�G�?���@���A?�a A�xLAM`C��@��@��BC��6A�h+C�(�A-+0C�A0�TB��Ax�A��A�;9A��Z               D                  G   ?      #   0   M            9      0      $               $   ;      Q   -      !                      2      	            4   	                        
               	                  Q               %   9   ?         '   3            %      %      /   /   #         )   =   =   /   3      #            %         #                  =         '                                                   M                  )   )            #                        /      #         )   =   9      -                  %                           1         '                                    N��N6��N4�8N.ԟP�9UOV�~N�fN� XN1�O`�[P��P�?N� LO��OV��O��EO!�N:yOOd"�NWw�O��OO-{;P2�NO�7�OKmO�+�O���Pt�Po:�O�ٯP9owN5l�O�@�O-��N&��N��O�#OQ}gN�GOA�N���N
v�N��IO=CyO1��P;N�N�XM���O�R\N��Nd�NO@N���N\X&NY
N� �O<�8N�%�O-CIN���N��"      �  u  �  �  �  �  F  �  p  Q  �  v  K  �    1  �  4  o  I  �  �  #  �  i  Z  �  �  (  	�  !  ]  �  �  �  �  x  �  G    �  �  6  �  I  �  �  L  }  �  Z  �  �  h  �  [    V  l  �  `<�h<�9X<�t�<D��;��
�o�o�t��D���T����/���u�49X��`B��P�#�
�#�
�49X�t��D�����ͼ��
��1�#�
��j�ě���/��j��j��/�ixռ������t��������\)�+�t��u�,1�'49X�8Q�8Q�T���@��D���D���m�h�Y��]/�aG��e`B�ixսm�h�y�#��o������T���T��������������������������������������������������#/50/#���#<U�����{I0����
#3<BECB=</#��������������������������������-0<IU^USI<20--------#*36CKOPPLC6*GKS^o��������zmaTNHG������	"�������������������������������������������������")5@BFHHGB@5)��������������������7;HTZ[ajdaTTH;62/277������

����������� ����<BHN[gnttrkg[NB>989<zz����������|zzzzzzz^z������������zna[X^����
#%(%#
�����o����������������xooY[glt������tg[ZWYYYY;HTWagp{���zaTJ@:88;�������������������������� ����������v{���������������zvv��������&62������#Unz����w`UH<,#*"'#����

���������)Bhtyyth[O)������������������������)45BN[ceee_SNB53+%%)��������������������������������������������������������������� #*$
�������5<@BO[ehie`[ZLB86515nqpu{�����������{ynnggst����������tggdgg��������������������ehtx}���tthdeeeeeeee����������������������������������������������������	)Nt���������fbN)	���������|}��������9<HIJJHD<;9999999999�����������������yz�#0010/(#��

����������������������������!#+/8<HB=</&##!!!!!!#'*),,#QUajnqnaYUROQQQQQQQQ!! ��������������������[anz�������znda[[[[[LNT[git������tg[ZRNL')55BNONGBA95)(&''''U[]gmt�����wtg[TUUUU²²¦¥¦ª²¿��������������¿²²²²���������ĽнؽٽнĽ��������������������������������������������������ìèçêìùú��ÿùìììììììììì�	�����g�S�C�>�.�A�s������� ����"�$�	�)�'�%�%�)�/�6�B�O�[�h�j�k�^�^�[�O�B�6�)�����������������������Ⱦž��������������׾ҾʾƾʾԾ׾޾����������׾׾׾׼������������żʼҼҼʼ¼����������������.��	�����������	��"�.�5�:�<�<�<�;�.�5�(����������5�N�\�r�������g�A�5������ �	��/�=�T�m�z���������z�T�H�"��M�H�A�M�Q�Z�f�s�������s�s�f�Z�M�M�M�M���������������������$�.�8�<�6�0�$�������s�i�h�s�u�����������������������������x�l�_�[�b�s���������ûлӻӻλû����������������������	�������	��������ɺ��������������ɺֺ׺ֺԺ˺ɺɺɺɺɺɻ������������!�-�:�A�:�8�0�/�-�!����������������������ĿɿѿҿɿĿ�������������������������������������������������j�^�W�V�Y�b�g�s�������������������������׾ӾʾǾžɾʾ׾�������������y�`�N�K�P�y�����Ŀѿ��ڿٿ�׿ǿ����y�����ݾ�����	����	�	�������ĿĦĖĦķ���������
��������������ĿàÓÑÏÓàåìù����������������ùìà����������������*�2�F�K�>�6�%������ɺ����r�Y�N�N�c�r�������ֺ��������ɼ��f�\�^�o�{�����ּ��!�.�8�9�!���ּ����H�8�)�'�5�H�T���������������������m�a�H�Y�M�A�.�+�4�@�Y�f�r���������������r�Y���s�j�j�r�z�|���������������������������Z�N�S�Z�g�j�s�t�s�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�����������*�6�C�O�S�V�Q�O�C�6�*�àØÓÑÊÇÂ�|ÁÇÓàèìðùûùìà�\�\�U�\�h�u�w�w�u�h�\�\�\�\�\�\�\�\�\�\�B�=�6�)�!�)�6�6�6�B�D�E�B�B�B�B�B�B�B�B�ѿĿ����������Ŀƿѿݿ�����������ݿѿ"��	���������	��.�D�G�T�^�T�N�G�;�"�@�4�'����
���'�.�4�7�@�K�M�N�M�@�@�������������!�)�)�/�/�*�)����[�X�[�[�h�tāćčĖčā�t�h�[�[�[�[�[�[�����'�3�3�4�4�3�,�'���������Y�T�M�@�4�@�B�M�V�Y�[�f�i�k�f�\�Y�Y�Y�Y�.�$�!������������!�'�7�:�E�?�B�9�.�>�=�@�G�K�H�L�W�e�q�v�w�x�u�r�e�Y�L�B�>ĦĚĒ�x�r�sāĚĳĿ��������������ĺĳĦ�������ʾ׾۾׾־ʾ���������������������F$FF#F$F1F=FDF=F6F1F$F$F$F$F$F$F$F$F$F$���x�g�a�^�]�_�i�x�}�~�������ûл������������������ûɻлܻ����ܻлû�������E*EE(E*E7E@ECEPEREZEPECE7E4E*E*E*E*E*E*�0�/�#�����#�*�0�6�<�@�<�0�0�0�0�0�0E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��ݽֽнĽ��Ľнݽ���������ݽݽݽݽݽ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��ݽӽ۽ݽ������������ݽݽݽݽݽ��0�,�$� ��)�0�=�I�V�b�o�t�o�k�b�V�I�=�0���������������ĿɿѿҿҿԿѿϿĿ���������������������������� �&�)�)�)����a�^�U�U�O�T�U�U�a�l�n�z�}�{�z�n�a�a�a�a�<�3�/�'�#��!�#�/�<�F�H�I�L�M�H�<�<�<�< ( ) B K h 9   R s D [ M = D ; > ! C C  h H J B , g + $ x b J D C 7   o A M J Q / A \ Y 5 N ` S = c g = s y ( d 3 ` A U  W P    �  I  S  b  �  �  �  �  �  �  �  �  �  !  �  �  Y  ]  C  �  �  �  |  1    �  �     �  �    2  s  A    �  /  ;  a  �  �  <  �  J  �  �  �  �      �  �  �  �  �  �  h  �  �  7  q  �    D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�     	           �  �  �  �  �  �  {  a  D  '  
  �  �  �                    	      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  2  �  �  k     �  �  +  �  /  �   �    *  G  Y  i  x  �  �  �  �  �  �  �  �  m  Y  C  ,    �  �  �  �  �  i  M  2  �  U  �  {  S  P  H    �  �  @  �   �  �  �  �  �  �  �  g  H  $  �  �  �  v  J    �  �  w  �  I  �  �  �  �  �  �  �  �  �  �  �  �  {  m  \  L  ;  (      �  (  S  t  �  �  �  �  �  �  �  q  0  �  �  )  �  �  �  �  F  D  A  >  <  :  ;  <  <  =  >  >  ?  ?  ?  =  :  6  3  /  �  �  �  �  �  �  �  �  �  �  �  �  y  Y  *  �  �  K  �  C  �  N  �  �  5  V  b  i  T  :  /    �  �  _  �  l  �  �  �  �  �  #  3  9  3  9  C  P  D  N  +  �  �  5  �  3  {  �  6  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  9  �  �     �  U  m  u  r  d  N  3    �  �  v  4  �  �  1  �  X  �  M  �  �  $  B  =  )      !  J  G  =  (    �  l    �    k  _  �    J  g  r  p  s  �  �  ~  S  0      �  `  �    �  Z    �  �  �  �  �  �  �  �  �  �  �  �  �  s  c  S  E  7  *  1  $      �  �  �  �  �  �  q  Y  A  )      �  �  �    �  �  �  �  x  l  g  a  L  5       �  �  �  �  Y  +  �  �  m  �    `  �  �    '  3  ,    �  �  }  -  �  .  �  �  !  o  l  j  g  e  d  d  d  g  p  y  �  s  U  7    �  �  �  V     -  C  H  I  F  <  )  	  �  �  w  9  �  �     �    I   �  r  �  �  �  �  �  �  �  y  g  R  ;  "    �  �  �  a    �  �  �  �  z  �  �  �  �  k  X  H  2    �  �  �  e    �  C  �  �  k  �  �  �  �  �  �      #  !    �  �  b  �  �  �  �  �  ~  �  b  :    �  �  p  .  �  �  S  �  �    �  �  <  b  h  c  V  E  2      �  �  �  �  �  �  k  E    4  �  �  C  S  Y  Z  V  L  ;        �  �  �  T     �  �  F  �  �  �  �  �  �  �  �  z  x  U  2  �  �  �  �  ^  �  �  u  2  �  *  �  �  �  Y  2    �  �  �  t  T  2    �  u    �        !  "    �  �  �  �  p  J  !  �  �  �  e  5    �  �  �  G  	.  	�  	�  	�  	�  	�  	�  	�  	�  	�  	w  	!  �  :  �  ,  s  3  �  P           �  �  �  �  b  W  X  P  6    �  �  4  �    p  ]  S  J  @  6  +        �  �  �  �  �  �  �  �  �  |  r  �  �  �  �  �  �  �  �  �  �  �  {  R  "  �  �  7  �  �  F  �  �  �  �  �  v  W  4    �  �  r  7  �  �  �  T  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  [  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	  
S  l  w  u  n  ^  F  %  �  �  �  X  )  '  �  �  Y  �  x  �  �  �  �  �  �  �  q  Z  A  %    �  �  �  �  �  �  j  ;     �  G  =  3  (      �  �  �  �  �  �  �  �    p  b  ^  Y  T  )  �  �  �  Z  �  �        �  �  �  \  �  �    �    *  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      8  �  �  �  �  �  �  �  �  �  �  �  s  G    �  �  �  M    �  /  .  .  4  .        	  �  �  �  �  Q  "  �  �  �  %  �  �  �  �  �  �  y  g  T  ?  )      �  �  �  �    �  �  �  I  8  (      �  �  �  �  �  �  _  @  #          3  g  s  �  �  �  �  l  �  �  u  `  C    �  �  A  �  �  4  �  �  �  �  �  �  �  �  s  X  ?  %    �  �  �  |  a  K  8  (    L  B  8  .  $        �  �  �  �  �  �  �  n  W  @  (    }  l  R  /      �  �  w  <  �  �  �  >  �  �  q  ^  9  �  ]  {  �  �  �  �  �  �  �  �  �  �  ~  ^  7    �  �  ,  �  �  >  I  /    �  �  �  :  �  �  [    �  c  	  �  @  �  y  �  �  �  �  �  �  �  �  �  }  v  o  i  c  \  [  \  ]  ^  _  �  �  �  k  M  +    �  �  g  &  �  �  L    �  �  q  :    h  ^  S  I  >  4  *  $  !               �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  g  M  1    �  �  �  �  [  R  I  @  1  !                         �  �  �      �  �  �  �    I    �  �  �  �  j  7  �  �  �  l  �  V  J  =  -      �  �  �  �  �  u  W  8    �  �  �  )  �  j  c  d  g  i  l  j  \  X  O  :       �  �  �  J  �  m  �  �  �  �  �  �  z  w  s  X  ;    �  �  �  �  b  :    �  �  `  P  >  *    �  �  �  �  Z  -  �  �  �  q  H  #  �  �  �