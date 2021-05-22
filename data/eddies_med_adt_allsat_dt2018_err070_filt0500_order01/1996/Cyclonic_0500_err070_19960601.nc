CDF       
      obs    U   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�ȴ9Xb     T  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�8   max       PEG�     T      effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��^5   max       <�t�     T   T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @F"�\(��     H  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v��Q�     H  .�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q            �  <8   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�          T  <�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <�o     T  >8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B/)�     T  ?�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B/�     T  @�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C���     T  B4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�@�   max       C���     T  C�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Z     T  D�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5     T  F0   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5     T  G�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�8   max       PCK�     T  H�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��1&�y   max       ?��~���%     T  J,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ȴ9   max       <�t�     T  K�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��
   max       @F!G�z�     H  L�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���
=p    max       @v��G�{     H  Z   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q            �  gd   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�          T  h   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B9   max         B9     T  id   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�t�j~��   max       ?���Fs��     �  j�               1            
         ?   #                     Y               3   
                  /                  $      $               +               !      4   	            
                     !      	      
               (                  1            	      NI(�NE*N��NelcO��lNG�!N٨wN�*Nб�Oo�uO��PBP�>O�oN��N�8N'��O�O�W`O��O���N}�O	�O�M�PEG�NFO�OGOӮzN��O	#�PCK�Oa��N��vN�8=O�M�O�T�O��N�i{O��[Os\/O/1~N#�O'�P�]Nlc�N��NO}9N��O��XN�ЂO��CN���N�P}N<G�O�N�0N�)�O��LN��O�K�N��IN�s�O�P�N��N��<O"��O')RN��Otu�NR�N�hO�	xN�x%N��N�8N���O*O���O�P�O!4�NAI�NZ�7N��N㘖<�t�<#�
<o<o;ě�;ě�;�o;D��;o;o;o:�o$�  ��o��o��o�ě���`B�o�o�#�
�D���T���T���e`B�u�u��o��o��C���t���t����㼴9X��9X��j��j��j��j���ͼ�����`B��h�������o�o�o�+�+�C��C��C��C��\)������w�#�
�#�
�,1�,1�0 Ž0 Ž0 Ž8Q�<j�<j�H�9�L�ͽL�ͽT���Y��Y��Y��aG��aG��ixսy�#��7L��7L��C���^5��^5STamoxqmjaTSSSSSSSSS���������������������������������������������������~��������
/HUZaa^UH/#��  
Xaknz����znaXXXXXXXX��������������������"/;<=;4/.&""""""""""������������������

�������#/:<@EB?</#!��������
����������� ����������*5CN[gu����vgNB;=88*
#/9;6/#
#)*##(/<H></##!")6;HOSY\hjaH1/.%"���������������������������
������������5BFNSSNL@5)������������������������������������������&1BOht����xh[OB6)%"&��������������;BOR[\[OB@;;;;;;;;;;��6BEJKJC?6*����|�����������������~|*<Un{�����nUIB=*12'*��������������������MOZ[chrpjnnh[OODCFMM�������(-+"�������5BNQTNLB5)	�
 "
	NO[^ehlttth[WOONNNNNbgmz���������zmaXUWb��������������������EIQUbnz{{zvpnaUHC>BE~������������~~~~~~~x���������������zrrxSYamz��������zmaTRQS#0<IKPSMI?<0*#������������������������������������������
0UblttobUI0
���z{{���������{zzzzzzJNV[ggtuxyvtga[SNLJJ|�����������������|bhkt�����vthg]bbbbbbVat������������tg[TV������� 	����������#3DMUanzyvnaU</#zz���������zyzzzzzz")266666865.)������������������������!!�����������������������������������������������������������������@BNUQNLB?>@@@@@@@@@@��������������������\gt���������tjg\\\\\��������������������[grs��������tg[USU[�������������}BBBIOUW[][SOJB;:?=BB$)6BO[hmmjhd[OB=3,)$��������������������fnv{��������{zniffff ')16873-*!��� Z[^gmt{{xtg[ZZZZZZZZ}�������������{z{{}}^anz���������znaZXW^����������������������������������������������������������������������������������������

����������tg][^pt���������$&)'# �����������


������������

�������������UUanuzona`WUUUUUUUUU
!! 
�!#%/2<HLPTPH</%#!!!!�I�G�@�G�I�U�b�c�b�\�W�U�I�I�I�I�I�I�I�I�z�t�r�zÇÒÓÛÓÇ�z�z�z�z�z�z�z�z�z�z¦¬²¸¿��¿²¦������#�-�/�9�/�)�#��������ūŨŬŵŶŹ����������������������Źū�U�N�H�H�C�B�H�U�U�Z�\�X�U�U�U�U�U�U�U�U�H�H�>�<�7�<�H�T�U�V�a�i�n�u�n�l�a�U�H�H�������������������������������������������������������)�2�)�)���������A�>�(� ���,�A�J�Z�^�f�n�o�a�^�Z�S�M�A�<�/�1�<�@�H�U�a�n�q�y�t�n�a�U�H�<�<�<�<���������������	��0�I�V�]�_�^�P�=�0��N�U�s�������������������������������s�N�A�5�4�(�'�5�M�Z�f�s��������������s�Z�A�}�t�l�g�b�`�g�tìéæìù��������ùìììììììììì�����������������������������������������������������������	��"� �/�3�"��	�����H�;�/�(�#�"�$�,�;�H�T�m�z�������z�W�HàÓÇÁÂÌÓàìù��������������ùìà�����������~�������������������������������������������������ûлѻлϻû�������øìàÜÖÖ×Ýàìôù������������ùø�����߾߾����� ��0�@�G�S�S�=�.�	���m�T�>�.���.�T�y���������ÿ��������y�m�a�W�]�a�m�n�o�y�q�n�a�a�a�a�a�a�a�a�a�a�G�F�'���׾׾���	�.�G�U�]�g�i�b�X�T�G�	��
�	�����	��"�*�1�/�/�(�"���	�-�*�#�#�-�:�S�l�x���{�t�z�x�l�_�S�F�:�-�y�o�m�`�_�[�Y�`�m�y�������������������y�ù����������ùϹܹ������������ܹϹùüʼ�����{�����ֽ��!�.�8�:�0�&����ּ��g�[�Z�Q�N�L�N�W�Z�g�{���������}�v�s�l�g�$���$�0�=�I�M�M�I�=�0�$�$�$�$�$�$�$�$�Y�L�T�Y�e�i�r�~�~�~�����~�r�n�e�Y�Y�Y�Y������������(�5�A�Q�Z�`�e�d�N�A�5��A�5�4�7�?�N�Z�g�s�������������s�g�Z�I�A�5�(������(�A�N�Z�g�j�u�w�w�s�g�A�5�'�����'�3�@�L�V�L�A�@�3�'�'�'�'�'�'��������ĻįħĦĳĺ����������������ƳƧƜƕƏƎƉƎƖƜƳ����������������Ƴ�����������������������������������������@�;�8�@�L�W�Y�[�Y�L�@�@�@�@�@�@�@�@�@�@�������������Ľǽнݽ����� ����ݽĽ�����������o�n�y�������Ľѽڽ˽ĽýĽ������������������#�%���������������������������ĿƿǿĿ������������ѿ˿ɿ̿Կݿ���������������ݿ�����¿²³¿���������������������������������������������������7�A�F�<�)������������������������������������������FFFFF$FBFVFeFmF|F�F�F�F�FlFZFCF$FFFJFIF=F5F5F=FJFVF[FcFcFcF\FVFJFJFJFJFJFJ�M�A�@�4�)�'��'�3�4�4�@�J�M�Y�c�[�Y�M�M�����߹�������������������������ŠŖŔŏŎœŔřŠŭŲŹ����ŹŷŮŭŠŠ�����������������������������������h�a�\�f�h�tāčĚĦİĦĚĖčĆā�t�h�h�������������������ûлܻ��ܻԻʻû����/�-�(�/�<�E�H�P�H�<�/�/�/�/�/�/�/�/�/�/�T�H�;�/�%�%�%�/�H�T�a�d�k�s�r�n�o�m�a�T��������������	����!����	���������	����	��"�$�.�1�2�.�"��	�	�	�	�	�	�"����������*�6�C�O�U�\�`�c�\�O�C�6�"ŹŷůŭŹ����������������������ŹŹŹŹ�����������������ûȻлһл˻û�������������~�~������������������������������������ֺƺĺɺֺ�������!�"� �������4�-�'�%�!�'�/�4�@�D�M�O�M�M�H�@�4�4�4�4�����нͽнԽݽ����(�4�=�<�.����������������!�&���������ùõëìù����������������������������ù�������������������׾��������پʾ��������	��� ���$�0�3�0�)�$������������8�=�@�I�V�b�o�{ǈǍǔǗǔǈ�{�o�b�V�I�8��ڼּϼʼʼʼ˼ͼּּ���������_�]�S�F�C�=�:�3�:�F�S�_�l�n�q�r�l�l�_�_������ֺɺź��������Ǻʺֺ��������������������¿¹®®±¿�����������������!�:�G�l�u�~���}�l�G�.�!����(�#���(�5�A�N�Z�c�g�l�g�a�Z�N�G�A�5�(�������������������������������������������ĿѿѿҿѿĿ�����������������EEEEEE*E7ECEKELECE7E*EEEEEEEEE D�D�D�D�D�D�D�EEEEEEEEEEE L 6 O ? 2 � 5 5 F 1 < 6 Z ? j , r ` D $ 7 V : < C U j t K Q / O S : b = / F @ D C & . L \ T / 2 @ V J ~ 0 Y \ A v j " X 3 Z # = $ Z W R = U D P $ V p ^ 1 y $ ` 2 , < ? -    t  K  �  w  �  �  �  C  �  �  -  �  �    �  ,  �  �  �  �  �  �  l    \  S  �  �    $  .  �    �  �  n  C  K  �  Z    r  3  �  �  �    D  �  g  %  �  �  �  `  P  g  #  #  )     A  �  �  �  �  �  �  �  2  {  ;  �  ;  v  )    �    *  `  N  z  �  <�o;D��;o;o�,1$�  �D���D����`B�#�
��o�y�#�\)������C��t��t���9X��`B�Ƨ��h�e`B�C��t���o�ě���������w���ͽ�㽅��C���h��P�0 Ž,1�m�h�C��y�#�,1�+�t��49X��t��t��8Q�0 Ž�w��+�'�{�0 Ž'��@��@��P�`��O߽,1�T���aG��D�������P�`�T���}�aG��]/��7L�u�m�h��j�q�������7L��C����-��
=��j��Q콑hs��������`A���B �\B3dB.%B.B��B �wA���BφB�BB#wB�1B�&B�_B��B@�A�P�B��B͉B��B*�nB�B��BJB��B/)�B e�B'��B*��BL�B-x�B��B�B_�A��<B��B�B/B��A��tB%�B"aB!��B&�B)DzB	�B/�BH�B
g�B{&B�WB��B=jB!�B1�B�B��B��BBh�B	��B��B	��B
��B�'B��B�VB(�@B�B	]�B
��BlB�=B
��B�GB!��B#�`B
�qB��B�B$�Bi B��B�A��7B ��B/�BA�B��B�%B ��A���B��B�FB?NB@�B�B�oB�)B��Bi�A��2B�{B�B��B+�B�5B�gB�B�VB/�B z�B'�B*�B?�B-�B�cB>6B@6A��|B�ZB=�B?�B��A��B&2wB!�YB!�B'0�B)EB�xB@oBSUB
;�B��B;$B��B;�B!բB(�B8�BɾBT�B�B8B	ÒBÎB	��B?�B��B��B��B(ǾB�vB	F�B
�BF�B4�B
�XB�KB!��B#�B
��B��B�wB$hBu�B�BхA�6�A�q�A���A�Y�A��,A�=#AŒ�A�%�A�AlA::�A�1B	�0A�:4A@ĳA��PA�E�A�,A�$�A�A͓WA�/�@��A̗�A\�}Aj�nA��A^�A�%�@���Al�>���A��A��hB
j]?�A��1A�)A��i?���A�T�Bp+A���?�?:A)��A b�A�"�Au�A~c�A���A�pA�^NC��C���@��?A�6A���A�{A��@@��A�x�A��AZ��A]��B Z�A��z@�D�A�W�@O��@�j�A1Y3A�ĦA϶�AN�B�`B��Ad�@�[@8D�A�֤A3pA���@�^MAxJC���C�TxA�)hA�~SA�BA�y�A��rAŅ"A�w�A�rUA�f�A:r�AŎ�B
 bA��LAAy�A�[XA�cA�kaA��^A��À�A�m�@��Ȃ�A[�AjA.A�vbA]L�A�m@{�Am"�>�@�A�.A��QB
�?�)/A�{5A�cA�~�?�O=A�B�A��A?�ߘA)1A H�A��AAt�A��A�r�A��{A�w C��-C���@Ӏ�?B��A�A�* A܋�@���A�ۍA�s�AZ��A]9�B ��A��@�wOA���@U7e@�"�A0�,A��A΃RAM��B�`Bv�A6H@�@4 <A�|�AS�A�fW@�\�Axh�C��vC�W�               1            
         ?   $                     Z               3   
            	      0                  $      %               +               "   	   4   
            
                     !   	   
      
   	         	   )                  1             	                                          '   +   %            )      !            #   /      )      '         5                        '               )               '      )                                                                                 !   !                                                      %               %                     %      '               5                        %               %                                                                                                      !   !               NI(�NE*N��N#l�O��lNG�!N٨wN�*N��N���N]<POuλO�:O�9N.�8N�8N'��O��MOxw�O%�O>A N}�O��O�,O�V�NFO���O�O�z�N�0NB�PCK�OK�WN��vNc�OT��O8\OUҧN�i{O��rOI��O/1~N#�O'�O���Nlc�N��NO}9N��O�nlN�ЂO`QNf��NX��N<G�O�N�0N�ܕO@N��O�K�N���N�s�O*��N��N��<O��O')RN��Otu�NR�N�hO�	xN��&N�|:N�8Nǵ�N��dO���O�P�O!4�NAI�NZ�7Nd�dN㘖  [  S  J  X    <  �  s  �  w  �    o  �  �  �  �  �  ]  �  �  �  �  r    �  �  ^      <  �  �  �  �    �  0  P  �  �    �  o  �  C  G  �  �    �  	�  n     J  �  #  w  �  �  �  !  u  ?  �  �  "  	  �  	  �  �  �  6  c  �  �    �  T     �  �  n  v<�t�<#�
<o;�`B;ě�;ě�;�o;D��$�  ��o�ě��ě���`B�u�o��o�ě��o�T���@���o�D���u��t����u��o��C���1�����/��t����
��9X��j��`B��h����j������`B��`B��h���\)���o�o�o�#�
�+�<j�\)�\)�C��\)���'8Q�#�
�#�
�0 Ž,1�Y��0 Ž0 Ž<j�<j�<j�H�9�L�ͽL�ͽT���]/�]/�Y��e`B�q���ixսy�#��7L��7L��C��ȴ9��^5STamoxqmjaTSSSSSSSSS������������������������������������������������������������
/HUZaa^UH/#��  
Xaknz����znaXXXXXXXX��������������������"/;<=;4/.&""""""""""������

����������������


����������!#,/6<<<8/&#!!!!!!!���������������������	�����������BN[bgkrtutjg[QNHFFBB#+/160/##)*##(/<H></##!#*;HLTXZffaTH;/("  #����������������������������������������)5?BFFB>85)����������������������������������������)+/:BO[h|}thbVOB6-))��������������;BOR[\[OB@;;;;;;;;;;��	*6CHJIC@<6*����������������������;?<BIbnw�����{nUI;;��������������������NOU[hjhh`[UONNNNNNNN�������(-+"������5;BNPRNIB5)
 "
	NOP[^ehlsh[XPONNNNNN]fmz|������zsma\Z[]��������������������HNUanquvwvrnaUHEDDEH~������������~~~~~~~x���������������zssxTTZamz������zma[TSRT#0<IKPSMI?<0*#�����������������������������������������
0UbiqqmbUI0
����z{{���������{zzzzzzJNV[ggtuxyvtga[SNLJJ|�����������������|bhkt�����vthg]bbbbbbY]ht�����������tgb]Y������� 	����������#/<FJLOU\`UH</*#��������|{����������$)36763-)������������������������!!������������������������������������������������������������������@BNUQNLB?>@@@@@@@@@@��������������������^gt�������tlg^^^^^^^��������������������[[`gt���������tg[[Y[�������������}BBBIOUW[][SOJB;:?=BB16BO[hklihc[OHB>65.1��������������������fnv{��������{zniffff ')16873-*!��� Z[^gmt{{xtg[ZZZZZZZZ}�������������{z{{}}^anz���������znaZXW^������������������������������������������������������������������������������������

���������������tg][^pt���������$&)'# �����������


������������

�������������UUanuzona`WUUUUUUUUU	

								!#%/2<HLPTPH</%#!!!!�I�G�@�G�I�U�b�c�b�\�W�U�I�I�I�I�I�I�I�I�z�t�r�zÇÒÓÛÓÇ�z�z�z�z�z�z�z�z�z�z¦¬²¸¿��¿²¦������#�)�/�5�/�&�#��������ūŨŬŵŶŹ����������������������Źū�U�N�H�H�C�B�H�U�U�Z�\�X�U�U�U�U�U�U�U�U�H�H�>�<�7�<�H�T�U�V�a�i�n�u�n�l�a�U�H�H���������������������������������������������������	���$�%����������4�-�(�'�(�(�4�?�A�B�M�M�Q�M�I�A�4�4�4�4�<�8�<�>�H�T�U�V�a�i�a�[�U�H�<�<�<�<�<�<�$������ ���#�0�=�I�J�P�M�I�F�=�0�$�s�k�c�s�������������������������������s�M�F�D�M�T�Z�f�o�s�t�������}�s�f�Z�M�M�w�t�i�g�f�g�tìéæìù��������ùìììììììììì���������������������������������������������������������	��� ��%���	�������;�/�,�)�,�5�;�E�H�T�a�m�z�|�q�l�`�K�H�;àÝÓÓÙàìù������������������ùìà�������������������������������������������������������������ûлѻлϻû�������àÝ×ØÙàãìïù����������ÿùìàà�"�	������������&�.�;�@�G�F�;�.�"�m�`�T�;�-�'�.�;�T�m�y���������������y�m�a�W�]�a�m�n�o�y�q�n�a�a�a�a�a�a�a�a�a�a�T�G�"��	���پ���	�*�;�G�Q�Y�e�f�_�T���	������	��"�"�)�/�0�/�.�%�"��S�F�:�-�*�$�&�-�:�F�S�_�l�v�w�r�i�m�_�S�y�s�m�c�`�`�`�m�y���������������y�y�y�y�Ϲιù����ùϹйܹ�ܹйϹϹϹϹϹϹϹϼʼ�����{�����ֽ��!�.�8�:�0�&����ּ��g�e�Z�S�O�M�N�P�Y�Z�g�w���������|�s�h�g�$���$�0�=�I�M�M�I�=�0�$�$�$�$�$�$�$�$�e�_�Y�U�Y�e�j�r�~�����~�r�i�e�e�e�e�e�e�(��������(�5�A�N�V�Z�]�X�N�A�5�(�N�A�<�<�A�G�N�Z�g�s�����������u�s�g�Z�N�3�(�����(�5�A�N�Z�c�n�p�l�g�Z�N�A�3�'�����'�3�@�L�V�L�A�@�3�'�'�'�'�'�'��������ļİĩĬĳĽ����������������ƳƱƧƞƗƓƕƚƧƳ������������������Ƴ�����������������������������������������@�;�8�@�L�W�Y�[�Y�L�@�@�@�@�@�@�@�@�@�@�������������Ľǽнݽ����� ����ݽĽ����������r�q�y���������˽ҽǽ��������������������������#�%���������������������������ĿƿǿĿ������������ѿ˿ɿ̿Կݿ���������������ݿ�����¿²³¿�����������������������������������������������)�/�8�<�=�0�)������������������������������������������FFFFF$F1F=FJFVFcFoF{FrFeFTFJF>F$FFF=F7F6F=FJFVFZFaFVFJF=F=F=F=F=F=F=F=F=F=�M�C�@�4�,�4�9�@�F�M�Y�a�Y�X�M�M�M�M�M�M�����߹�������������������������ŠŖŔŏŎœŔřŠŭŲŹ����ŹŷŮŭŠŠ�����������������������������������h�d�_�h�r�tāČčĐčĂā�t�h�h�h�h�h�h���������������������ûȻлڻջлŻû����/�-�(�/�<�E�H�P�H�<�/�/�/�/�/�/�/�/�/�/�T�H�;�/�%�%�%�/�H�T�a�d�k�s�r�n�o�m�a�T������������	��� ����	�������������	����	��"�$�.�1�2�.�"��	�	�	�	�	�	�6�5�*�����*�1�6�C�D�O�T�Z�\�O�M�C�6ŹŷůŭŹ����������������������ŹŹŹŹ�����������������ûȻлһл˻û���������������������������������������������������ֺƺĺɺֺ�������!�"� �������4�-�'�%�!�'�/�4�@�D�M�O�M�M�H�@�4�4�4�4�����нͽнԽݽ����(�4�=�<�.����������������!�&���������ùõëìù����������������������������ù�������������������׾��������پʾ�������������� �
���������������������V�K�I�D�I�V�b�o�{ǈǌǔǗǔǈ�{�o�b�V�V��ڼּϼʼʼʼ˼ͼּּ���������F�E�?�<�F�S�_�l�m�p�r�l�_�S�F�F�F�F�F�F�ɺȺ��������ɺֺ�����ֺɺɺɺɺɺ���������������¿¹®®±¿�����������������!�:�G�l�u�~���}�l�G�.�!����(�#���(�5�A�N�Z�c�g�l�g�a�Z�N�G�A�5�(�������������������������������������������ĿѿѿҿѿĿ�����������������EEEEE*E7ECEGEHECE7E*EEEEEEEEEE D�D�D�D�D�D�D�EEEEEEEEEEE L 6 O : 2 � 5 5 C ) K - Y 7 a , r b H ' * V 9 : @ U h m A 5 B O O : V K $ : @ D 3 & . L \ T / 2 @ ( J z 3 G \ A v >  X 3 Z # % $ Z I R = U D P $ ` i ^ / A $ ` 2 , < 6 -    t  K  �  D  �  �  �  C  �  �  |  �  9  3  x  ,  �  f    n  �  �  =  o    S  #  h  H  �  5  �  �  �  z  �  �  �  �  E  �  r  3  �  h  �    D  �  S  %  9  b  a  `  P  g  �  �  )       �  f  �  �  W  �  �  2  {  ;  �  �  1  )  �  �    *  `  N  z  y    B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  B9  [  U  O  I  C  =  7  1  *  $    
  �  �  �  �  �  �  �  �  S  J  A  8  -  #    
  �  �  �  �  �  �  �  �  �  �  �  �  J  <  .         �  �  �  �  �  �  �  �  ~  o  \  1    �  T  U  V  W  S  M  G  <  .     	  �  �  �  �  ]  7     �   �      �  �  �  �  s  @    �  D  C  $  �  �  `    �  �  h  <  ?  C  F  F  E  D  B  >  :  6  0  +    �  �  P  /    �  �  �  �  �  �  �  {  i  U  @  *    �  �  �  m  3  �  -  ~  s  c  S  B  /      �  �  �  �  �  �  }  n  _  Q  F  :  /  w  �  �  �  �  �  �  �  �  �  �  {  o  S    �  �  b  5    4  7  ;  >  E  N  [  e  n  t  t  m  a  P  8    �  �  �  &  �    &  K  x  �  �  �  �  �  �  �  t  J    �  �    �  0  �    R  �  �  �  �       �  �  �  �  R  �  �    *  �  p  /  K  ^  k  o  j  ^  I  2  /  0    �  �  �  J  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  5  �  �  �  �    >  \  x  �  �  �  �  �  �  �  �  �  �    Z  �    �  �  �  �  �  �  �  �  �  �  �  �  w  h  Y  J  9  (      �  �  �  �  �  �  �  �  ~  h  S  =  '    �  �  �  �  �  p  {  �  �  �  }  q  `  M  :  +  $        �  �  x  >   �   �  L  N  Q  W  [  \  Y  V  O  F  ;  4  *    �  �  �  e  I  1  �  �  	Z  	�  
~  
�  H  y  �  �  s  K    
�  
$  	C  �  2    �  l    �  �  �  �  �  �  �  �  }  g  N  +    �  �  S  ?    �  �  �  �  |  w  r  m  h  c  _  ]  Z  X  U  S  P  N  K  I  �  �  �  �  �  �  m  M  (     �  �  i  ,  �  �  .  �  X  �  I  Y  e  o  r  j  \  I  0    �  �  �  Y  1  �  �  P   �   �  �  �  �  �  �  �    �  �  �  �  �  a  /  �  �    y  �  �  �  }  r  e  W  J  <  .  !    �  �  �  �  z  K    �  �  w  �  �  �  �  �  �  �  �  �  �  �  x  W  8  
  �  �  �  :   �  3  J  ^  [  W  N  C  3  "    �  �  �  �  x  B  �  �  _  	  	            �  �  �  �  �  �  ]  1  	  �  �  n  T  �  o  h  `  f  t  {  m  `  K  5    �  �  �  �  u  I  "     �  *  -  *  &  "      %  9  =  8  (    �  �  b    �  P  �  �  �  l  L  /    �  �  �  c  9    �  �  L    �  U  �   �  �  �  �  �  �  �  t  a  K  0    �  �  �  }  :  �  �  <   �  �  �  x  m  a  T  D  1      �  �  �  �  v  V  6    �  �  p  �  �  y  k  U  9      �  �  v  6  �  �  t  .  �  �  �  �  �    	        �  �  �  �  s  A    �  y  (  �     x  �  �  �  �  �  �  �  �  �  �  �  �  h  7    �  �  m  =    �    )  .  0  -  %    �  �  �  �  O  
  �  ^  �  ]  B   �  P  B  2  !    �  �  �  �  �  �  b  A    �  �  �  �  \  2  �  �  m  F    �  �  ]    �  �  �  �  ~  q  ,  �  u  �    e  t  �  �  v  c  L  ,    �  �  �  V  $  �  �  �  i  ;                        
        �   �   �   �   �   �   _  �  �  �  �  �  �  �  �  �  �  �  }  v  o  h  `  X  O  E  <  o  k  d  [  O  B  2      �  �  �  l  8    �  �  `      �  �  �  �  �  �  �  p  ]  c  F  "      �  �  Z  �  c  x  �  C  >  :  5  /  "    	  �  �  �  �  �  �  �  s  ^  I  3    G  ?  4  &    �  �  �  �  ~  \  8    �  �  �  h  5  �  �  �  �  �  �  |  m  W  @  %  
  �  �  �  �  p  N  +    �  �  �  �  �  |  j  Y  F  3      �  �  �  �  �  �  �  W     �  �  �  �  �      �  �  �  �  �  x  H    �  e    �  I  p  �  �  �  �  �  �  �  y  j  Z  L  @  3  '    �  �  �  $   �  	O  	{  	�  	�  	�  	�  	�  	b  	6  	�  	V  	  �  a    �  �  �  c  %  f  j  m  g  `  X  P  F  =  2  '        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  Z  6    �  �  �  h  G  J  ;  ,      �  �  �  �  �  �  �  d  H  +  	   �   �   �   w  �  �  �  �  �  �  w  f  S  <  $  
  �  �  �  o  6  �  �    #        �  �  �  �  �  j  M  8  ,  $  $  !    	  �  �  �    K  j  s  u  o  [  D  )    �  �  r  8  �  �  m    �  7  O  j  �  �  �  �    l  T  6    �  �  S  �  �  .  �  p  �  }  v  o  h  a  Z  S  L  E  @  ;  7  3  /  +  &  "      �  �  �  }  o  `  P  =  +  $    �  �  �  �  �  `  3   �   �          
  �  �  �  �  �  �  R    �  �  1  �  o  	   �  u  r  o  l  e  ]  U  J  =  /      �  �  �  �  �  �  f  L     (  #        >  -    �  �  �  �  �  _    t  �  �   �  �  �  �  �  �  r  ^  I  1    �  �  �  �  �  y  O      �   �  �  �  �  v  l  v  �  �  n  \  J  8  &      �  �  �     [  �        
  �  �  �  �  �  �  r  E    �  �  [  '  �  |  	  �  �  �  �  �  �  v  Y  <    �  �  �  �  d  M  2     �  �  �  �  �  �  �  �  x  `  I  1      �  �  �  �  �  �  �  	  �  �  �  �  �  �  �  m  O  :  %    �  �  �  �  �  �  �  �  �  �  �  r  \  F  /      �  �  �  �  �  c  E  '    �  �  w  Y  ;      �  �  �  x  H    �  �  �  {  S  (   �   �  �  k  \  F  &    �  �  �  x  N    �  �  _    �  7  �  V  1  3  4  6  ,        �  �  �  �  �  �  �  �  �  V  %   �  P  \  [  G  1      �  �  �  �  �  �  �  q  J    �  �  |  �  �  �  �  n  P  2    �  �  �  �  q  N  ,  
  �  �  �  R  �  �  �  �  �  �  u  k  [  F  (    �  �  �  p  E    �  �    d  �      �  �  �  �  �  t  E    �  �  ;  �  �  P  �  �  �  |  Z  7    �  �  �  �  g  /  �  �  P  �  Y  �  �  �  T  A  )    �  �  �  ]  2  !  C  N  =    �  �  J  �  �  *       	  �  �  �  �  �  l  4  �  �  j    �  {  6  )    �  �  {  v  q  l  f  ^  W  O  G  @  9  2  +  $    	  �  �  �  �  �  �  �  �  �  �  �  |  m  \  K  8  %    �  �  �  �  �  /  '  (  ;  W  e  m  \  ;    �  �  �  �  h  D     �  �  �  v  o  h  ^  S  F  6  %    �  �  �  �  �  �  �  z  p  q  w